"""Regression coverage for the local coder's Java-only response boundary."""
import unittest

from minecraft_mod_ai.construction_file_author import (
    ConstructionFileError,
    _only_java,
)


class ConstructionFileAuthorProtocolTest(unittest.TestCase):
    def test_raw_java_is_preserved(self):
        source = "package example;\npublic class Example {}\n"
        self.assertEqual(_only_java(source), source)

    def test_fenced_java_is_unwrapped(self):
        source = "```java\npackage example;\npublic class Example {}\n```"
        self.assertEqual(
            _only_java(source),
            "package example;\npublic class Example {}\n",
        )

    def test_fake_tool_wrapper_is_not_executed(self):
        source = (
            "I will write the file. "
            "<tool_call><function=write><parameter=content>"
            "package example;\npublic class Example {}"
            "</parameter><parameter=path>/tmp/Example.java</parameter>"
            "</function></tool_call>"
        )
        self.assertEqual(
            _only_java(source), "package example;\npublic class Example {}\n"
        )

    def test_tool_only_response_cannot_pass_as_java(self):
        text = (
            "<tool_call><function=bash><parameter=command>"
            "ls src/main/java"
            "</parameter></function></tool_call>"
        )
        with self.assertRaisesRegex(
            ConstructionFileError, "CONSTRUCTION_FILE_NON_JAVA_RESPONSE"
        ):
            _only_java(text)

    def test_mixed_markdown_cannot_pass_as_java(self):
        with self.assertRaisesRegex(
            ConstructionFileError, "CONSTRUCTION_FILE_EXTRA_CONTENT"
        ):
            _only_java("package example;\n```java\nclass X {}\n```")


if __name__ == "__main__":
    unittest.main()
