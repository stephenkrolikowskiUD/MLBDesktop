import ast
from pathlib import Path
import unittest
from urllib.parse import parse_qs, urlsplit


class ScheduleScopeTest(unittest.TestCase):
    def test_live_schedule_does_not_exclude_postseason(self):
        source = Path(__file__).with_name("MLBEnginev5-4.py").read_text()
        tree = ast.parse(source)
        assignments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "sched_url"
                    for t in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        expression = ast.Expression(assignments[0].value)
        url = eval(compile(expression, "schedule_url", "eval"), {
            "MLB_API": "https://statsapi.mlb.com/api/v1",
            "schedule_date": "2026-10-03",
        })
        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query["sportId"], ["1"])
        self.assertEqual(query["date"], ["2026-10-03"])
        self.assertEqual(query["hydrate"], ["probablePitcher,team,venue"])
        self.assertNotIn("gameType", query)
        self.assertNotIn("gameTypes", query)


if __name__ == "__main__":
    unittest.main()
