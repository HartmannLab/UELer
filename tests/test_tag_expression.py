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


class TagExpressionSpacedNameTests(unittest.TestCase):
    """Issue #130: tags containing spaces must work without quoting."""

    def test_bare_spaced_tag(self):
        predicate = compile_tag_expression("my tag")
        self.assertTrue(predicate(["my tag"]))
        self.assertFalse(predicate(["my", "tag"]))

    def test_spaced_tags_with_operators(self):
        predicate = compile_tag_expression("tumour core & !necrotic edge")
        self.assertTrue(predicate(["tumour core", "stroma"]))
        self.assertFalse(predicate(["tumour core", "necrotic edge"]))
        self.assertFalse(predicate(["stroma"]))

    def test_spaced_tags_with_parentheses(self):
        predicate = compile_tag_expression("good roi & (figure 1 | figure 2)")
        self.assertTrue(predicate(["good roi", "figure 2"]))
        self.assertFalse(predicate(["good roi", "figure 3"]))
        self.assertFalse(predicate(["figure 1"]))

    def test_leading_and_trailing_spaces_are_stripped(self):
        predicate = compile_tag_expression("   my tag   &   other tag   ")
        self.assertTrue(predicate(["my tag", "other tag"]))
        self.assertFalse(predicate(["my tag"]))

    def test_internal_whitespace_runs_are_collapsed(self):
        predicate = compile_tag_expression("my    tag")
        self.assertTrue(predicate(["my tag"]))
        self.assertTrue(predicate(["my\ttag"]))

    def test_tag_whitespace_is_normalised_too(self):
        predicate = compile_tag_expression("my tag")
        self.assertTrue(predicate(["  my  tag  "]))

    def test_quoted_and_bare_spaced_tags_are_equivalent(self):
        bare = compile_tag_expression("my tag | other")
        quoted = compile_tag_expression("'my tag' | other")
        for tags in (["my tag"], ["other"], ["unrelated"]):
            self.assertEqual(bare(tags), quoted(tags), tags)

    def test_quoting_still_allows_operator_characters(self):
        predicate = compile_tag_expression("'a&b' & c")
        self.assertTrue(predicate(["a&b", "c"]))
        self.assertFalse(predicate(["a", "b", "c"]))

    def test_adjacent_quoted_names_report_missing_operator(self):
        with self.assertRaises(TagExpressionError) as ctx:
            compile_tag_expression("'my tag' 'other tag'")
        self.assertIn("Missing operator", str(ctx.exception))

    def test_empty_quoted_name_raises(self):
        with self.assertRaises(TagExpressionError):
            compile_tag_expression("'' & alpha")

    def test_whitespace_only_expression_raises(self):
        with self.assertRaises(TagExpressionError):
            compile_tag_expression("   ")

    def test_unclosed_quote_still_raises(self):
        with self.assertRaises(TagExpressionError):
            compile_tag_expression("'my tag & other")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
