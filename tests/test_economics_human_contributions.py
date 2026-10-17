"""Import original human manuscripts without creating LLM resolution claims."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.economics_catalog import (BASE_DIR, HUMAN_CONTRIBUTIONS_PATH,
                                      build_economics_data, generate_economics_data)
from scripts.validate_submissions import validate_file


class EconomicsHumanContributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = dict(id='C73-1', initial_rank=1, title='Example', jel_code='C73',
                           source_id='OP-0001',
                           statement_file='attacks/open_problems/economics/statements/C73-1.tex')
        self.write('lists/economics/problems.json', json.dumps(dict(
            schema_version=1, problem_count=1, problems=[self.record])))
        self.write('lists/economics/rankings.csv', 'Problem ID,New rank\nC73-1,1\n')
        self.write('lists/economics/jel_codes.json',
                   (BASE_DIR / 'lists/economics/jel_codes.json').read_text())
        self.write(self.record['statement_file'],
                   '% BEGIN ECONOMICS STATEMENT\n\\subsection{Question}\nOriginal target.\n'
                   '% END ECONOMICS STATEMENT\n')
        self.relative = 'attacks/open_problems/economics/Human_Contribution/C73-1.tex'
        self.original = ('% ECONOMICS_PROBLEM: C73-1\n'
                         '\\documentclass{article}\n\\title{Original manuscript}\n'
                         '\\begin{document}\\maketitle\nA claimed full proof.\n'
                         '\\section{Completion Estimate}\n100\\%\n'
                         '\\section*{References}\nOriginal source.\n\\end{document}\n')
        self.write(self.relative, self.original)
        self.entry = dict(problem_id='C73-1', filename='C73-1.tex', author='Example Author',
                          submitted_by='contributor', first_posted='2026-10-12',
                          submission_url='https://github.com/mehmetmars7/Erdosproblems-llm-hunter/pull/176',
                          original_source='https://github.com/author/paper/tree/' + 'a' * 40,
                          tex_sha256=hashlib.sha256(self.original.encode()).hexdigest(),
                          scope='A declared restricted theorem.', ai_use='AI-assisted proof work.',
                          verification='Author checked; independent review pending.',
                          review_status='submitted')
        self.manifest([self.entry])

    def write(self, path, text):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)

    def manifest(self, entries):
        self.write(HUMAN_CONTRIBUTIONS_PATH, json.dumps(dict(
            schema_version=1, contributions=entries)))

    def validate(self):
        return validate_file(self.root, self.relative,
                             {'economics': {'C73-1': self.record}})

    def test_import_preserves_original_and_separates_claim_from_review(self):
        self.assertIn('human contribution', self.validate())
        problem = generate_economics_data(self.root)['C73-1']
        attack, = problem['attacks']
        self.assertEqual(attack['entry_kind'], 'human_contribution')
        self.assertEqual(attack['review_status'], 'submitted')
        self.assertEqual(attack['author'], 'Example Author')
        self.assertEqual(problem['status'], 'open')
        self.assertEqual(problem['llm_status'], 'none')
        self.assertNotIn('completion', problem)
        self.assertNotIn('completion', attack)
        self.assertEqual((self.root / self.relative).read_text(), self.original)
        self.assertEqual((self.root / 'docs' / attack['download_url']).read_text(), self.original)
        self.assertFalse(list((self.root / 'docs').rglob('*.pdf')))

    def test_changed_original_is_rejected_by_builder_and_validator(self):
        self.write(self.relative, self.original.replace('full proof', 'different proof'))
        for check in (self.validate, lambda: build_economics_data(self.root)):
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                check()

    def test_unregistered_human_manuscript_does_not_bypass_validation(self):
        (self.root / HUMAN_CONTRIBUTIONS_PATH).unlink()
        with self.assertRaisesRegex(ValueError, 'registered provenance'):
            self.validate()
        with self.assertRaisesRegex(ValueError, 'Unregistered Economics human'):
            build_economics_data(self.root)

    def test_unsafe_or_unreviewed_metadata_is_rejected(self):
        changes = [('filename', '../C73-1.tex'), ('problem_id', 'C73-999'),
                   ('review_status', 'accepted'), ('original_source', 'https://github.com/a/b/tree/main'),
                   ('submission_url', 'javascript:alert(1)'), ('author', '')]
        for key, value in changes:
            with self.subTest(key=key):
                entry = {**self.entry, key: value}
                self.manifest([entry])
                with self.assertRaises(ValueError):
                    build_economics_data(self.root)

    def test_duplicate_records_are_rejected(self):
        self.manifest([self.entry, copy.deepcopy(self.entry)])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            build_economics_data(self.root)

    def test_version_ten_is_a_distinct_preserved_original(self):
        path = 'attacks/open_problems/economics/Human_Contribution/C73-1_v10.tex'
        self.write(path, self.original)
        self.manifest([self.entry, {**self.entry, 'filename': 'C73-1_v10.tex'}])
        self.assertEqual([a['version'] for a in build_economics_data(self.root)['C73-1']['attacks']],
                         [1, 10])

    def test_normal_llm_claims_retain_their_existing_status(self):
        metadata = {key: self.record[key] for key in ('id', 'title', 'jel_code', 'source_id')}
        source = ('% ECONOMICS_PROBLEM: ' + json.dumps(metadata) + '\n'
                  '% FIRST_POSTED: 2026-10-12\n% ATTEMPT_STATUS: solved\n'
                  + self.original.split('\n', 1)[1])
        self.write('attacks/open_problems/economics/gpt_6_astra_pro/C73-1.tex', source)
        problem = build_economics_data(self.root)['C73-1']
        self.assertEqual(problem['status'], 'solved')
        self.assertEqual(problem['llm_status'], 'solved')
        self.assertEqual(problem['completion'], 100)


if __name__ == '__main__':
    unittest.main()
