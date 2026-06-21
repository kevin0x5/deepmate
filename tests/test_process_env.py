import unittest

from deepmate.runtime.process_env import subprocess_environment


class ProcessEnvironmentTests(unittest.TestCase):
    def test_subprocess_environment_normalizes_ascii_locale(self) -> None:
        env = subprocess_environment(
            {
                "LANG": "C",
                "LC_ALL": "C.UTF-8",
                "LC_CTYPE": "POSIX",
            }
        )

        self.assertEqual(env["LANG"], "en_US.UTF-8")
        self.assertEqual(env["LC_ALL"], "en_US.UTF-8")
        self.assertEqual(env["LC_CTYPE"], "en_US.UTF-8")

    def test_subprocess_environment_preserves_utf8_locale(self) -> None:
        env = subprocess_environment(
            {
                "LANG": "zh_CN.UTF-8",
                "LC_ALL": "zh_CN.UTF-8",
                "LC_CTYPE": "zh_CN.UTF-8",
            }
        )

        self.assertEqual(env["LANG"], "zh_CN.UTF-8")
        self.assertEqual(env["LC_ALL"], "zh_CN.UTF-8")
        self.assertEqual(env["LC_CTYPE"], "zh_CN.UTF-8")


if __name__ == "__main__":
    unittest.main()
