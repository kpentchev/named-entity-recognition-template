import nltk
from bs4 import BeautifulSoup
import urllib.request
import csv

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

urlpages = [
    #'https://esportsobserver.com/echo-fox-opseat-sponsor/',
    #'https://esportsobserver.com/lol-eu-minor-leagues/',
    #'https://esportsobserver.com/china-recap-feb6-2019/',
    #'https://esportsobserver.com/overwatch-league-brazil-broadcast/',
    #'https://esportsobserver.com/slg-best-buy-logitechg-challenge/'
    #'https://esportsobserver.com/hyperx-marco-reus-brand-ambassador/',
    #'https://esportsobserver.com/bird-bird-legal-support-esic/',
    #'https://esportsobserver.com/single-player-games-twitch/',
    #'https://esportsobserver.com/fortnite-marshmello-concert/'
    #'https://esportsobserver.com/globe-telecom-mineski-team-liyab/',
    #'https://esportsobserver.com/blizzard-owl-coca-cola/',
    #'https://esportsobserver.com/newzoo-ceo-esports-phase-two/'
    #'https://esportsobserver.com/psyonix-eleague-rocket-league/',
    #'https://esportsobserver.com/esl-paysafecard-through-2019/',
    #'https://esportsobserver.com/hyperx-pittsburgh-knights-partner/',
    #'https://esportsobserver.com/psg-mobile-legends-team/'
    'https://www.forbes.com/sites/mattperez/2019/02/26/g2-esports-raises-17-3-million-in-series-a-funding'
    ]

rows = []
sentenceIdx = 131
for urlpage in urlpages:
    page = urllib.request.urlopen(urlpage)
    soup = BeautifulSoup(page, 'html.parser',)
    soup.prettify('UTF-8')
    #print(soup)
    
    text = ' '.join( x.getText() for x in soup.find('div', {'class': 'article-container'}).find_all('p'))
    #text = ' '.join( x.getText() for x in soup.find('div', {'class': 'entry-content'}).find_all('p'))
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").replace("…", "...").replace("–", "-")
    #print(text)

    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        first_row_sentence = [f'Sentence {sentenceIdx}']
        for val in tagged[0]:
            first_row_sentence.append(val)
        rows.append(first_row_sentence)
        for row in tagged[1:]:
            a_row = ['']
            for val in row:
                a_row.append(val)
            rows.append(a_row)
        sentenceIdx += 1
            #print(tagged)
            #print('\n-----\n')
    #print('\n-----\n'.join('\t'.join(x) for x in rows))

with open('teo_tagged_clean.csv', 'w', encoding='UTF-8') as output:
    csv_output = csv.writer(output, delimiter='\t')
    csv_output.writerows(rows)
    



