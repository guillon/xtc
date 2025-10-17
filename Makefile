check: check-type check-lit-all check-pytest

check-type: check-pyright check-mypy

check-pyright:
	pyright

check-mypy:
	mypy

check-lit-all:
	$(MAKE) check-lit
	$(MAKE) check-lit-c

check-lit:
	lit -v tests/filecheck

check-lit-c:
	env XTC_MLIR_TARGET=c lit -v tests/filecheck/backends tests/filecheck/mlir_loop

check-pytest:
	scripts/pytest/run_pytest.sh -v

.PHONY: check check-lit check-lit-c check-pytest check-type check-pyright check-mypy
.SUFFIXES:
