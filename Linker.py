from neo4j import GraphDatabase

class Linker(object):

    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Connected to neo4j database at {}".format(uri))

        self._ner2link = {
            'gam': _linkGame,
            'mon': _linkMoney
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

'''
CALL db.index.fulltext.queryNodes("labelIdx", "\'CS\\:GO\'") YIELD node, score 
MATCH (g:Game) -[:HAS_LABEL]-> (node)
MATCH (t:Tournament) -[r:COMPETITION_FOR] -> (g) 
WHERE t.date > '2017-12-31' AND t.date < '2019-01-01'
MATCH (p:Player) -[tp:TournamentParticipation]-> (t) 
RETURN g, SUM(tp.earning) AS prize_money, COUNT(distinct p) AS num_players, COUNT(distinct t) AS num_tournaments, MAX(score) AS s ORDER BY s DESC
'''
def _linkGame(con, text):
    queryTemplate = '''
                    CALL db.index.fulltext.queryNodes("labelIdx", "\'{}\'") YIELD node, score
                    MATCH (g:Game) -[:HAS_LABEL]-> (node)
                    MATCH (t:Tournament) -[r:COMPETITION_FOR] -> (g)
                    MATCH (p:Player) -[tp:TournamentParticipation]-> (t)
                    RETURN g, SUM(tp.earning) AS prize_money, COUNT(distinct p) AS num_players, COUNT(distinct t) AS num_tournaments, MAX(score) AS s ORDER BY s DESC
                    '''

    result = con.run(queryTemplate.format(text.replace(':', '\\:')))
    row = result.single()
    game = row['game']
    prize_money = row['prize_money']
    num_players = row['num_players']
    num_tournaments = row['num_tournaments']
    return {
        'name': game['name'],
        'synonyms': game['altNames'],
        'prize_money': prize_money,
        'num_players': num_players,
        'num_tournaments': num_tournaments
    }

def _linkMoney(con, text):
    text = text.split(' ')[0]
    currency = text[0:1]
    numeric = text[1:].replace('M', '000000').replace('K', '000')
    return {
        'currency': currency,
        'value': float(numeric)
    }
