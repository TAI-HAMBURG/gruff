import csv

from pronouns import pronoun_mapping
from article import article_mapping
from occupations import occupation_mapping
from pairings import iter_paired_variants

with open('data/task.tsv', 'r',
          encoding='utf-8') as input_file, \
        open('tasks_no_context.tsv', 'w', newline='', encoding='utf-8') as output_file:
    reader = csv.DictReader(input_file, delimiter='\t')
    writer = csv.writer(output_file, delimiter='\t')

    writer.writerow([
        'occupation', 'participant', 'sentence', 'pronoun_type',
        'word', 'article_and_occupation', 'pronoun', 'uid', 'confuse_pronoun'
    ])

    # Daten verarbeiten
    for row in reader:
        pronoun_type = row['pronoun_type']
        pronouns = pronoun_mapping[pronoun_type]
        occupation_type = row['occupation']
        occupations = occupation_mapping['occupation'][occupation_type]
        articles = article_mapping
        # Schreibe fuer jede Kombination aus Occupation, Article und Pronoun eine Zeile
        for _, pronoun, article, occupation in iter_paired_variants(pronouns, occupations, articles):
            writer.writerow([
                occupation,  # occupation
                row['participant'],  # participant
                row['sentence'].replace('$ARTICLE_AND_OCCUPATION', f"{article} {occupation}"),  # sentence
                row['pronoun_type'],  # pronoun_type
                row['word'],  # word
                f"{article} {occupation}",  # article_and_occupation
                pronoun,  # pronoun
                row.get('uid', ''),  # uid
                ''  # confuse_pronoun (leer)
            ])
