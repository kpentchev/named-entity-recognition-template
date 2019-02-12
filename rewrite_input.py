import csv

with open('/home/kpentchev/data/floyd/ner_2019_02_11.csv', encoding='utf-8') as csv_read:
    csv_reader = csv.reader(csv_read, delimiter='\t')
    line_count = 0
    sentence_count = 1
    with open('/home/kpentchev/data/floyd/ner_2019_02_11_fixed.csv', encoding='utf-8', mode='w') as csv_write:
        csv_writer = csv.writer(csv_write, delimiter='\t')
        for row in csv_reader:
            if line_count == 0:
                csv_writer.writerow(row)
            else:
                sentence = row[0].strip()
                if (not "" == sentence):
                    row[0] = f"Sentence {sentence_count}"
                    sentence_count += 1
                csv_writer.writerow(row)
            line_count += 1

