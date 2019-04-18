import nltk
from bs4 import BeautifulSoup
import urllib.request
import csv
from tqdm import tqdm

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
    #'https://www.forbes.com/sites/mattperez/2019/02/26/g2-esports-raises-17-3-million-in-series-a-funding',
    #'https://esportsobserver.com/rainbow-six-pro-league-rev-sharing/',
    #'https://esportsobserver.com/nissan-faze-clan-optic-gaming/',
    #'https://esportsobserver.com/csgo-pro-league-france/',
    #'https://esportsobserver.com/t1-faceit-apex-legends/',
    #'https://esportsobserver.com/aquilini-funding-luminosity-gaming/',
    #'https://www.forbes.com/sites/mikeozanian/2018/10/23/the-worlds-most-valuable-esports-companies-1/#48a2b6a06a6e',
    #'https://www.forbes.com/sites/forbessanfranciscocouncil/2019/03/11/five-esports-predictions-what-does-the-year-hold-for-companies-and-developers/#18d3edde395b',
    'http://www.espn.com/esports/story/_/id/26130966/astralis-build-counter-strike-legacy-iem-katowice-title',
    'http://www.espn.com/esports/story/_/id/26136799/where-rainbow-six-siege-goes-g2-pengu-follow',
    'http://www.espn.com/esports/story/_/id/26171562/checking-jensen-team-liquid',
    'http://www.espn.com/esports/story/_/id/26230906/take-look-fortnite-500000-secret-skirmish',
    'http://www.espn.com/esports/story/_/id/26249076/call-duty-franchise-spots-sell-25-million-per-team',
    'http://www.espn.com/esports/story/_/id/26265839/team-solomid-zven-smoothie-chat-mistakes-improvements',
    'http://www.espn.com/esports/story/_/id/26275659/from-scrims-stage-100-thieves-searches-same-page',
    'http://www.espn.com/esports/story/_/id/26298615/looking-griffin-surprising-loss-geng',
    'http://www.espn.com/esports/story/_/id/26352540/vancouver-titans-crush-overwatch-league-stage-1'
    ]

rows = []
sentenceIdx = 131
for urlpage in tqdm(urlpages):
    page = urllib.request.urlopen(urlpage)
    soup = BeautifulSoup(page, 'html.parser',)
    soup.prettify('UTF-8')
    #print(soup)
    
    text = ' '.join( x.getText() for x in soup.find('div', {'class': 'article-body'}).find_all('p'))   #espn
    #text = ' '.join( x.getText() for x in soup.find('div', {'class': 'article-container'}).find_all('p'))  #forbes
    #text = ' '.join( x.getText() for x in soup.find('div', {'class': 'entry-content'}).find_all('p'))      #teo
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").replace("…", "...").replace("–", "-")
    #print(text)

    print('Processing sentences for {}'.format(urlpage))
    sentences = tokenizer.tokenize(text)
    for sentence in tqdm(sentences):
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
    



