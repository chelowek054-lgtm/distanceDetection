from main import main


def test_main_rejects_visual_prompt_before_runtime_imports(capsys) -> None:
    exit_code = main(["--prompt-image", "reference.jpg"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Visual prompt requires a bbox provider" in captured.err

