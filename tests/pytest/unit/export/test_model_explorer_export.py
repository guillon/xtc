import json


def _build_matmul_graph():
    import xtc.graphs.xtc.op as O

    a = O.tensor((4, 8), "float32", name="A")
    b = O.tensor((8, 16), "float32", name="B")
    with O.graph(name="matmul_test") as gb:
        O.matmul(a, b, name="C")
    return gb.graph


def test_export_matmul_graph_structure():
    from xtc.export import graph_to_model_explorer

    graph = _build_matmul_graph()
    payload = graph_to_model_explorer(graph)

    assert payload["format"] == "xtc.model_explorer.v1"
    assert payload["name"] == "matmul_test"
    assert payload["inputs"] == list(graph.inputs)
    assert payload["outputs"] == list(graph.outputs)
    assert len(payload["nodes"]) == len(graph.nodes)
    assert len(payload["edges"]) == 2

    node = payload["nodes"][0]
    assert node["is_graph_input"] is False
    assert node["is_graph_output"] is True
    assert node["op"] == "matmul"
    assert node["input_types"] is not None
    assert node["output_types"] is not None


def test_export_can_disable_access_maps():
    from xtc.export import graph_to_model_explorer

    graph = _build_matmul_graph()
    payload = graph_to_model_explorer(graph, include_access_maps=False)
    for node in payload["nodes"]:
        assert "accesses_maps" not in node


def test_save_json_file(tmp_path):
    from xtc.export import save_model_explorer_json

    graph = _build_matmul_graph()
    out = tmp_path / "graph.model_explorer.json"
    save_model_explorer_json(graph, out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["name"] == "matmul_test"
    assert len(loaded["nodes"]) == 1
    assert len(loaded["edges"]) == 2
