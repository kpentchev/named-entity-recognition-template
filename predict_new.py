from NerModel import restore

import nltk

#text = 'The Berlin-based esports organization G2 Esports announced Tuesday it raised $17.3 million in a Series A funding round, bringing its total outside backing to $24.6 million. The round was led by three firms: Everblue Management, the family office of hedge fund manager Eric Mindich, who is also part of the G2\'s board; Parkwood Corporation, the family office of Cleveland\'s Mandel family; and Seal Rock Partners, the private equity vehicle for Jonathan and Edward Cohen, who serve as executives for oil and gas companies like Atlas Energy. Joining them are New York City real estate executives Al Tylis and Dan Gilbert; Brian Distelburger, the cofounder of data management tool Yext; Myke Naef, founder of the scheduling app Doodle; and Topgolf Media president YuChiang Cheng. "Our major goal was to put together an ownership group that would make our efforts even better of making this a billion-dollar company in a reasonable time frame," says G2 Esports cofounder and CEO Carlos Rodriguez. G2 Esports was founded in 2014 by Rodriguez, a former League of Legends professional player, and Jens Hilgers, founder of esports production company and tournament organizer ESL. Last year, Forbes named G2 Esports the eighth most valuable esports team in the world, worth an estimated $105 million. This month, Rodriguez was featured on Forbes\' 30 Under 30 in Europe list for Sports and Games. With the influx of capital, the team is eyeing global expansion to further take advantage of what\'s predicted to be a billion-dollar industry by the end of this year. "It\'s always been our goal to make this a Western organization, focused not only in Europe but also North America and to some degree South America," Rodriguez says.'
text = 'Starladder will present the 15th Counter-Strike: Global Offensive (CS:GO) Major, considered to be one of the most prestigious regular events in the game’s calendar. As well as being a first for the Ukraine-based esports company, this will be the first Major to take place in Berlin, from Sept. 5-8 at the Mercedes-Benz Arena.'
#text = 'Dota 2 developer Psyonix and Turner Sports’ ELEAGUE esports brand have announced a multi-faceted business, event, and broadcasting deal around the Dota 2 Championship Series (RLCS) and collegiate esports. ELEAGUE will produce a multi-part feature series around the upcoming RLCS Season 7, which begins its regular season in April, as well as the future Season 8 to follow. The series will cover big moments and behind-the-scenes stories from each season and will debut on the TBS cable network “later in 2019,” according to a release. Additionally, Turner Sports’ ad sales team will oversee all advertising and sponsorships for those two RLCS seasons. Furthermore, ELEAGUE and Psyonix will host a Collegiate Dota 2 (CRL) Spring Invitational event at the NCAA Final Four Fan Fest presented by Capital One, which will be held in Minneapolis from April 5-8. The top four teams from the CRL Spring Season will participate in the exhibition tournament.'
#text = 'ESL and Supercell reveal $ 1M Clash of Clans World Championship'
#text = 'The ESL Mobile Open will begin its first season on March 18 and feature three games: racer Asphalt 9: Legends, strategy game Clash of Clans, and PUBG Mobile, the smartphone and tablet adaptation of battle royale hit PLAYERUNKNOWN’S BATTLEGROUNDS. For season one, PUBG Mobile has a $53K USD prize pool, while Clash of Clans has $39K up for grabs, and Asphalt 9 players can vie for $13K in prizing.'

sentences = nltk.sent_tokenize(text)

model = restore('/Users/kpentchev/data/models/2019_04_05_16_37_stem_char_lstm_crf.h5')

for sentence in sentences:
    prediction = model.predict(sentence)

    for w, pred1 in prediction:
        print("{:15}: {:5}".format(w[0], pred1))
