"""Dependency-light source contract tests for the web UI."""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class WebUISourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "app.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.routes = {
            decorator.args[0].value
            for node in ast.walk(cls.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "app"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
        }

    def test_python_source_parses(self):
        self.assertTrue(self.tree.body)

    def test_required_api_routes_exist(self):
        for route in ("/api/health", "/api/generate", "/api/history", "/api/gallery", "/api/jobs/{job_id}"):
            with self.subTest(route=route):
                self.assertIn(route, self.routes)

    def test_frontend_is_persian_and_rtl(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('lang="fa" dir="rtl"', html)
        self.assertIn("گالری تصاویر", html)
        self.assertIn("پرامپت اصلی", html)

    def test_requirements_include_web_and_inference_runtime(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for dependency in ("fastapi", "uvicorn", "diffusers", "python-dotenv", "torch"):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, requirements)


if __name__ == "__main__":
    unittest.main()
