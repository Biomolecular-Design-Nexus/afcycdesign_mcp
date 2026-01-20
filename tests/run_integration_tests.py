#!/usr/bin/env python3
"""Automated integration test runner for Cyclic Peptide MCP server."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class MCPTestRunner:
    def __init__(self, server_path: str, env_path: str):
        self.server_path = Path(server_path).resolve()
        self.env_path = Path(env_path).resolve()
        self.project_root = self.server_path.parent.parent  # server.py is in src/, we want project root
        self.results = {
            "test_date": datetime.now().isoformat(),
            "server_path": str(self.server_path),
            "env_path": str(self.env_path),
            "project_root": str(self.project_root),
            "tests": {},
            "issues": [],
            "summary": {}
        }

    def test_server_startup(self) -> bool:
        """Test that server starts without errors."""
        try:
            cmd = ["env", f"PYTHONPATH={self.project_root}",
                   "mamba", "run", "-p", str(self.env_path), "python", "-c",
                   "from src.server import mcp; print('Server started successfully')"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.project_root
            )
            success = result.returncode == 0 and "Server started successfully" in result.stdout

            self.results["tests"]["server_startup"] = {
                "status": "passed" if success else "failed",
                "output": result.stdout.strip(),
                "error": result.stderr.strip(),
                "command": " ".join(cmd)
            }

            if not success:
                self.results["issues"].append({
                    "test": "server_startup",
                    "issue": "Server failed to import or start",
                    "details": result.stderr or result.stdout
                })

            return success
        except subprocess.TimeoutExpired:
            self.results["tests"]["server_startup"] = {
                "status": "failed",
                "error": "Test timed out after 30 seconds"
            }
            self.results["issues"].append({
                "test": "server_startup",
                "issue": "Server startup timed out"
            })
            return False
        except Exception as e:
            self.results["tests"]["server_startup"] = {"status": "error", "error": str(e)}
            self.results["issues"].append({
                "test": "server_startup",
                "issue": f"Test execution error: {e}"
            })
            return False

    def test_rdkit_import(self) -> bool:
        """Test that RDKit is available."""
        try:
            cmd = ["mamba", "run", "-p", str(self.env_path), "python", "-c",
                   "from rdkit import Chem; mol = Chem.MolFromSmiles('CCO'); print(f'RDKit OK: {mol is not None}')"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            success = result.returncode == 0 and "RDKit OK: True" in result.stdout

            self.results["tests"]["rdkit_import"] = {
                "status": "passed" if success else "failed",
                "output": result.stdout.strip(),
                "error": result.stderr.strip(),
                "command": " ".join(cmd)
            }

            if not success:
                self.results["issues"].append({
                    "test": "rdkit_import",
                    "issue": "RDKit not available or not working",
                    "details": result.stderr or result.stdout
                })

            return success
        except Exception as e:
            self.results["tests"]["rdkit_import"] = {"status": "error", "error": str(e)}
            self.results["issues"].append({
                "test": "rdkit_import",
                "issue": f"Test execution error: {e}"
            })
            return False

    def test_script_imports(self) -> bool:
        """Test that core scripts can be imported."""
        scripts = [
            "predict_cyclic_structure",
            "design_cyclic_sequence",
            "design_cyclic_binder"
        ]

        all_success = True
        for script in scripts:
            try:
                pythonpath = f"{self.project_root}:{self.project_root}/scripts"
                cmd = ["env", f"PYTHONPATH={pythonpath}",
                       "mamba", "run", "-p", str(self.env_path), "python", "-c",
                       f"import sys; sys.path.insert(0, 'scripts'); from {script} import run_{script}; print('{script} import OK')"]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.project_root
                )
                success = result.returncode == 0 and f"{script} import OK" in result.stdout

                self.results["tests"][f"{script}_import"] = {
                    "status": "passed" if success else "failed",
                    "output": result.stdout.strip(),
                    "error": result.stderr.strip()
                }

                if not success:
                    all_success = False
                    self.results["issues"].append({
                        "test": f"{script}_import",
                        "issue": f"Script {script} failed to import",
                        "details": result.stderr or result.stdout
                    })

            except Exception as e:
                all_success = False
                self.results["tests"][f"{script}_import"] = {"status": "error", "error": str(e)}
                self.results["issues"].append({
                    "test": f"{script}_import",
                    "issue": f"Test execution error: {e}"
                })

        return all_success

    def test_job_manager(self) -> bool:
        """Test that job manager is working."""
        try:
            cmd = ["env", f"PYTHONPATH={self.project_root}",
                   "mamba", "run", "-p", str(self.env_path), "python", "-c",
                   "from src.jobs.manager import job_manager; print(f'Job manager OK: {job_manager.jobs_dir.exists()}')"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.project_root
            )
            success = result.returncode == 0 and "Job manager OK: True" in result.stdout

            self.results["tests"]["job_manager"] = {
                "status": "passed" if success else "failed",
                "output": result.stdout.strip(),
                "error": result.stderr.strip()
            }

            if not success:
                self.results["issues"].append({
                    "test": "job_manager",
                    "issue": "Job manager not working properly",
                    "details": result.stderr or result.stdout
                })

            return success
        except Exception as e:
            self.results["tests"]["job_manager"] = {"status": "error", "error": str(e)}
            self.results["issues"].append({
                "test": "job_manager",
                "issue": f"Test execution error: {e}"
            })
            return False

    def test_claude_mcp_registration(self) -> bool:
        """Test that server is registered with Claude Code."""
        try:
            result = subprocess.run(
                ["claude", "mcp", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )

            success = result.returncode == 0 and "cycpep-tools" in result.stdout and "✓ Connected" in result.stdout

            self.results["tests"]["claude_mcp_registration"] = {
                "status": "passed" if success else "failed",
                "output": result.stdout.strip(),
                "error": result.stderr.strip()
            }

            if not success:
                self.results["issues"].append({
                    "test": "claude_mcp_registration",
                    "issue": "Server not properly registered with Claude Code",
                    "details": result.stdout + "\n" + result.stderr
                })

            return success
        except Exception as e:
            self.results["tests"]["claude_mcp_registration"] = {"status": "error", "error": str(e)}
            self.results["issues"].append({
                "test": "claude_mcp_registration",
                "issue": f"Test execution error: {e}"
            })
            return False

    def test_fastmcp_dev_server(self) -> bool:
        """Test that fastmcp dev server can start."""
        try:
            # Try to start server for 3 seconds to see if it initializes
            cmd = ["timeout", "3s", "mamba", "run", "-p", str(self.env_path), "fastmcp", "dev", str(self.server_path)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            # For fastmcp dev, a timeout is expected - we just want to see it starts without immediate errors
            # Check if it shows startup messages or if there are actual errors
            has_startup_indicators = any(indicator in result.stdout or indicator in result.stderr
                                       for indicator in ["Starting", "MCP inspector", "Proxy server", "listening"])
            has_real_errors = any(error in result.stderr.lower()
                                for error in ["traceback", "importerror", "modulenotfound", "syntaxerror"])

            # Success if we see startup indicators and no real errors, or just timeout (124)
            success = ((result.returncode == 124 or result.returncode == 0) and
                      (has_startup_indicators or not has_real_errors))

            self.results["tests"]["fastmcp_dev_server"] = {
                "status": "passed" if success else "failed",
                "output": result.stdout.strip(),
                "error": result.stderr.strip(),
                "note": "Timeout expected for dev server test",
                "return_code": result.returncode
            }

            if not success:
                self.results["issues"].append({
                    "test": "fastmcp_dev_server",
                    "issue": "FastMCP dev server failed to start",
                    "details": result.stderr or result.stdout
                })

            return success
        except Exception as e:
            self.results["tests"]["fastmcp_dev_server"] = {"status": "error", "error": str(e)}
            self.results["issues"].append({
                "test": "fastmcp_dev_server",
                "issue": f"Test execution error: {e}"
            })
            return False

    def generate_report(self) -> str:
        """Generate JSON report."""
        total = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"].values() if t.get("status") == "passed")
        failed = sum(1 for t in self.results["tests"].values() if t.get("status") == "failed")
        errors = sum(1 for t in self.results["tests"].values() if t.get("status") == "error")

        self.results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": f"{passed/total*100:.1f}%" if total > 0 else "N/A",
            "overall_status": "PASS" if failed == 0 and errors == 0 else "FAIL"
        }

        return json.dumps(self.results, indent=2)

    def run_all_tests(self):
        """Run all integration tests."""
        print("🧪 Running MCP Cyclic Peptide Tools Integration Tests")
        print("=" * 60)

        tests = [
            ("Server Startup", self.test_server_startup),
            ("RDKit Import", self.test_rdkit_import),
            ("Script Imports", self.test_script_imports),
            ("Job Manager", self.test_job_manager),
            ("Claude MCP Registration", self.test_claude_mcp_registration),
            ("FastMCP Dev Server", self.test_fastmcp_dev_server)
        ]

        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            try:
                success = test_func()
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"   {status}")
            except Exception as e:
                print(f"   💥 ERROR: {e}")

        # Generate summary before printing
        self.generate_report()

        print(f"\n📊 Test Summary:")
        print(f"   Total: {self.results['summary']['total_tests']}")
        print(f"   Passed: {self.results['summary']['passed']}")
        print(f"   Failed: {self.results['summary']['failed']}")
        print(f"   Errors: {self.results['summary']['errors']}")
        print(f"   Pass Rate: {self.results['summary']['pass_rate']}")
        print(f"   Overall: {self.results['summary']['overall_status']}")

        if self.results["issues"]:
            print(f"\n⚠️  Issues Found:")
            for i, issue in enumerate(self.results["issues"], 1):
                print(f"   {i}. {issue['test']}: {issue['issue']}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_integration_tests.py <server_path> <env_path>")
        print("Example: python run_integration_tests.py src/server.py ./env")
        sys.exit(1)

    server_path = sys.argv[1]
    env_path = sys.argv[2]

    runner = MCPTestRunner(server_path, env_path)
    runner.run_all_tests()

    # Save report
    report = runner.generate_report()
    report_path = Path("reports/step7_integration_tests.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)

    print(f"\n📝 Full report saved to: {report_path}")

    # Exit with appropriate code
    sys.exit(0 if runner.results["summary"]["overall_status"] == "PASS" else 1)

if __name__ == "__main__":
    main()