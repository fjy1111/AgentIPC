import agentipc

from agentipc.cli import main
from agentipc.config import AgentIPCConfig
from agentipc.utils import new_message_id, utc_timestamp


def test_core_package_smoke(capsys) -> None:
    assert isinstance(agentipc.__version__, str)
    assert agentipc.__version__

    config = AgentIPCConfig()
    assert config.llm_provider == "mock"
    assert config.embedding_provider == "hash"

    message_id = new_message_id()
    assert message_id.startswith("msg_")
    assert utc_timestamp() > 0

    exit_code = main(["version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert agentipc.__version__ in captured.out