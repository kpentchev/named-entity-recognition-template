from neo4j import GraphDatabase

class Linker(object):

    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Connected to neo4j database at {}".format(uri))

        self._ner2link = {
            'gam': _linkGame,
            'mon': _linkMoney
        }

        self._ner2query2 = {
            'geo': None,
            'gam': 'CALL db.index.fulltext.queryNodes("labelIdx", "\'{}\'") YIELD node, score MATCH (g:Game) -[:HAS_LABEL]-> (node) RETURN g, MAX(score) as s ORDER BY s DESC',
            'tea': 'Team',
            'tou': 'Tournament',
            'org': None,
            'tim': None,
            'mon': None
        }

    def getLink(self, text, etype):
        if etype in self._ner2link:
            con = self._driver.session()

            link_fun = self._ner2link[etype]
            return link_fun(con, text)
        else:
            return {}


    def close(self):
        print("Closing neo4j driver...")
        self._driver.close()

def _linkGame(con, text):
    result = con.run('CALL db.index.fulltext.queryNodes("labelIdx", "\'{}\'") YIELD node, score MATCH (game:Game) -[:HAS_LABEL]-> (node) RETURN game, MAX(score) as s ORDER BY s DESC'.format(text))
    rec = result.single()['game']
    return {
        'name': rec['name'],
        'synonyms': rec['altNames']
    }

def _linkMoney(con, text):
    text = text.split(' ')[0]
    currency = text[0:1]
    numeric = text[1:].replace('M', '000000').replace('K', '000')
    return {
        'currency': currency,
        'value': float(numeric)
    }
