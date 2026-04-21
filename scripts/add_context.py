import csv
import sys
import itertools
from pronouns import pronoun_mapping
from article import article_mapping
from participants import participant_mapping
from occupations import occupation_mapping
from pairings import iter_paired_variants
from pathlib import Path
import pandas as pd

def instantiate_template(template, article, referent, pronoun_type, pronoun):
    return template.replace('$ARTICLE', article).replace('$OCCUPATION/PARTICIPANT', referent).replace(pronoun_type, pronoun)

def build_pronoun_type_template_mapping(filename):
    pronoun_type_template_mapping = {
        'explicit_template': {pronoun_type: [] for pronoun_type in pronoun_mapping},
        'implicit_template': {pronoun_type: [] for pronoun_type in pronoun_mapping},
    }

    df = pd.read_csv(filename, delimiter='\t')
    for i, row in df.iterrows():
        pronoun_type = row['pronoun_type']
        polarity = row['polarity']
        for key in pronoun_type_template_mapping:
            pronoun_type_template_mapping[key][pronoun_type].append((row[key], polarity))

    return pronoun_type_template_mapping

def replace_task_sentence(row, article_and_occupation): #Replaces the $ARTICLE_AND_OCCUPATION in sentence and in article_and_occupation in task templates
    row = row.copy()
    row['sentence'] = row['sentence'].replace('$ARTICLE_AND_OCCUPATION', article_and_occupation)
    row['article_and_occupation'] = row['article_and_occupation'].replace('$ARTICLE_AND_OCCUPATION', article_and_occupation)
    return row

def get_output_line(row, context, pronoun1, uid, confuse=''):
    capitalized = [c[0].upper() + c[1:] if c else c for c in context] # Context bleibt unverändert - nur 1. Buchstabe capitalized
    template = ' '.join((*capitalized, row['sentence']))
    return '\t'.join([row['occupation'],
                      row['participant'],
                      template,
                      row['pronoun_type'],
                      row['word'],
                      row['article_and_occupation'],
                      pronoun1,
                      uid,
                      confuse,]) + '\n'

def add_context(filename, pronoun_type_template_mapping, occupation):
    f = 'o' if occupation else 'p' # first
    s = 'p' if occupation else 'o' # second
    basename = Path(filename).stem
    with open(filename, 'r', encoding='utf-8') as in_f, \
         open(f'e{f}_{basename}.tsv', 'w', encoding='utf-8') as ef_f, \
         open(f'e{f}_e{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_f, \
         open(f'e{f}_e{s}_i{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_is_f, \
         open(f'e{f}_e{s}_i{s}_i{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_is_is_f, \
         open(f'e{f}_e{s}_i{s}_i{s}_i{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_is_is_is_f, \
         open(f'e{f}_e{s}_i{s}_i{s}_i{s}_i{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_is_is_is_is_f:
        header = 'occupation\tparticipant\tsentence\tpronoun_type\tword\tarticle_and_occupation\tpronoun\tuid\tconfuse_pronoun\n'
        ef_f.write(header)
        ef_es_f.write(header)
        ef_es_is_f.write(header)
        ef_es_is_is_f.write(header)
        ef_es_is_is_is_f.write(header)
        ef_es_is_is_is_is_f.write(header)
        reader  = csv.DictReader(in_f, delimiter='\t')
        first = 'occupation' if occupation else 'participant'
        second = 'participant' if occupation else 'occupation'
        for row in reader:
            pronoun_type = row['pronoun_type']
            pronouns = pronoun_mapping[pronoun_type]
            occupation_type = row[first]
            occupations = occupation_mapping[first][occupation_type]
            participant_type = row[second]
            participants = participant_mapping[second][participant_type]
            occupation_variants = list(iter_paired_variants(pronouns, occupations, article_mapping))
            participant_variants = list(iter_paired_variants(pronouns, participants, article_mapping))
            for i, (e1, s1) in enumerate(pronoun_type_template_mapping['explicit_template'][pronoun_type]):
                for o, (_, pronoun1, article1, occupation_form) in enumerate(occupation_variants): #pronoun/article pairing is controlled via pairings.py
                    article_and_occupation = f"{article1} {occupation_form}"
                    intro1 = instantiate_template(e1, article1, occupation_form, pronoun_type, pronoun1) #früher war statt occupation row[first]
                    updated_row = replace_task_sentence(row, article_and_occupation)
                    ef_f.write(get_output_line(updated_row, [intro1], pronoun1, f'e{f}{i}'))

                    for j, (e2, s2) in enumerate(pronoun_type_template_mapping['explicit_template'][pronoun_type]):
                        if (j % 5) == (i % 5): # second template cannot have the same content as the first, regardless of polarity
                            continue
                        if s2 == s1: # use the opposite sentiment
                            continue
                        for p, (_, pronoun2, article2, participant_form) in enumerate(participant_variants): #pronoun/article pairing is controlled via pairings.py
                            if pronoun1 == pronoun2: # we need unique pronouns for each entity being spoken about
                                continue
                            if article1 == article2: continue
                            intro2 = instantiate_template(e2, article2, participant_form, pronoun_type, pronoun2) # früher war statt occupation row[second]
                            ef_es_f.write(get_output_line(updated_row, [intro1, intro2], pronoun1, f'e{f}{i}_e{s}{j}', pronoun2))

                            # implicit continuations must have the same sentiment and referent as the last intro
                            # it should not have the same content as either intro
                            # i.e., there should be 4 options
                            implicit_continuations = []
                            for k, (it, st) in enumerate(
                                    pronoun_type_template_mapping['implicit_template'][pronoun_type]):
                                if k == j or k == i:
                                    continue
                                if st != s2:
                                    continue
                                # must be filled with the same pronoun as the last intro because it is the same referent
                                implicit = instantiate_template(template= it, article=article2, referent = participant_form, pronoun_type=pronoun_type, pronoun = pronoun2)
                                implicit_continuations.append((k, implicit))
                            assert len(implicit_continuations) == 4

                            for perm in itertools.permutations(implicit_continuations, 1):
                                k1, i1 = perm[0]
                                ef_es_is_f.write(get_output_line(updated_row, [intro1, intro2, i1], pronoun1,
                                                 f'e{f}{i}_e{s}{j}_i{s}{k1}', pronoun2))

                            for perm in itertools.permutations(implicit_continuations, 2):
                                k1, i1 = perm[0]
                                k2, i2 = perm[1]
                                ef_es_is_is_f.write(get_output_line(updated_row, [intro1, intro2, i1, i2], pronoun1,
                                                 f'e{f}{i}_e{s}{j}_i{s}{k1}_i{s}{k2}', pronoun2))

                            # exploit the fact that perm(S, 3) == perm(S, 4) when |S| == 4
                            for perm in itertools.permutations(implicit_continuations, 4):
                                k1, i1 = perm[0]
                                k2, i2 = perm[1]
                                k3, i3 = perm[2]
                                k4, i4 = perm[3]
                                ef_es_is_is_is_f.write(get_output_line(updated_row, [intro1, intro2, i1, i2, i3], pronoun1,
                                                 f'e{f}{i}_e{s}{j}_i{s}{k1}_i{s}{k2}_i{s}{k3}', pronoun2))
                                ef_es_is_is_is_is_f.write(get_output_line(updated_row, [intro1, intro2, i1, i2, i3, i4], pronoun1,
                                                 f'e{f}{i}_e{s}{j}_i{s}{k1}_i{s}{k2}_i{s}{k3}_i{s}{k4}', pronoun2))

def main():
    assert len(sys.argv) == 3
    pronoun_type_template_mapping = build_pronoun_type_template_mapping(sys.argv[2])
    add_context(sys.argv[1], pronoun_type_template_mapping, occupation=True)

if __name__ == '__main__':
    main()
