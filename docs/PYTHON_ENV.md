# Python 환경 문제 (`_ctypes`, pyenv 3.14)

## 증상

```text
ModuleNotFoundError: No module named '_ctypes'
```

`import pandas` 직후 또는 표준 라이브러리 `ctypes` 로딩 시 발생합니다.

## 원인

**pyenv로 소스 빌드한 Python**이 시스템에 **`libffi`** 가 없을 때 `_ctypes` 모듈이 빠진 채로 설치되는 경우가 많습니다. 이 상태에서는 pandas·matplotlib 등 대부분의 데이터 스택이 동작하지 않습니다.

## 해결 (택 1)

### A. 시스템/다른 버전 Python 사용 (가장 빠름)

이 레포는 **Python 3.10**에서 검증했습니다.

```bash
/usr/bin/python3.10 -m pip install -r requirements.txt
/usr/bin/python3.10 pipeline_assignment_toolkit.py
```

### B. pyenv 3.14 재설치 (libffi 포함)

Ubuntu/Debian 예:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
pyenv uninstall -f 3.14.2
pyenv install 3.14.2
```

그 후 `pip install -r requirements.txt` 다시 실행합니다.

### C. 3.14 대신 3.12·3.13 pyenv 사용

데이터 과학 패키지 휠 호환성이 더 안정적인 경우가 많습니다.

## `ydata-profiling` (별도)

Python **3.14**에서는 `requirements.txt`의 환경 마커로 **설치 대상에서 제외**됩니다. 프로파일 전체 리포트가 필요하면 **3.10~3.13** 환경에서만 추가 설치하세요.
