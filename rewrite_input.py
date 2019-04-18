import csv

with open('/Users/kpentchev/data/ner_2019_04_06_prep.csv', encoding='utf-8') as csv_read:
#with open('/home/kpentchev/data/floyd/ner_2019_02_25_fixed.csv', encoding='utf-8') as csv_read:
    csv_reader = csv.reader(csv_read, delimiter='\t', quoting=csv.QUOTE_NONE)
    line_count = 0
    sentence_count = 1
    with open('/Users/kpentchev/data/ner_2019_04_06_fixed.csv', encoding='utf-8', mode='w') as csv_write:
    #with open('/home/kpentchev/data/floyd/ner_2019_02_25_fixed.csv', encoding='utf-8', mode='w') as csv_write:
        csv_writer = csv.writer(csv_write, delimiter='\t')
        for row in csv_reader:
            if line_count == 0:
                csv_writer.writerow(row)
            else:
                sentence = row[0].strip()
                if (not "" == sentence):
                    row[0] = f"Sentence {sentence_count}"
                    sentence_count += 1
                if len(row) > 4:
                    print('{} old {}'.format(line_count, row))
                    row = [row[0], ''.join(row[1:-2]).replace('"', '') ,row[-2] ,row[-1]]
                    print('{} new {}'.format(line_count, row))
                csv_writer.writerow(row)
            line_count += 1

