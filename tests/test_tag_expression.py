import unittest

from ueler.viewer.tag_expression import TagExpressionError, compile_tag_expression


class TagExpressionTests(unittest.TestCase):
    def test_conjunction(self):
        predicate = compile_tag_expression("alpha & beta")
        self.assertTrue(predicate(["alpha", "beta"]))
        self.assertFalse(predicate(["alpha"]))

    def test_disjunction(self):
        predicate = compile_tag_expression("alpha | beta")
        self.assertTrue(predicate(["alpha"]))
        self.assertTrue(predicate(["beta"]))
        self.assertFalse(predicate(["gamma"]))

    def test_negation(self):
        predicate = compile_tag_expression("!excluded")
        self.assertTrue(predicate(["alpha"]))
        self.assertFalse(predicate(["excluded"]))

    def test_parentheses_precedence(self):
        predicate = compile_tag_expression("alpha & (beta | gamma)")
        self.assertTrue(predicate(["alpha", "gamma"]))
        self.assertFalse(predicate(["gamma"]))

    def test_supports_quotes(self):
        predicate = compile_tag_expression("'alpha beta' & !delta")
        self.assertTrue(predicate(["alpha beta", "gamma"]))
        self.assertFalse(predicate(["alpha beta", "delta"]))

    def test_invalid_expression_raises(self):
        with self.assertRaises(TagExpressionError):
            compile_tag_expression("alpha & | beta")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
