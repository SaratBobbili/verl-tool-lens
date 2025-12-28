import ray


@ray.remote
class HRLDataSharingController:
    """Stores transcript and routing metadata per request."""

    def __init__(self) -> None:
        self._state: dict[str, dict] = {}

    def start(self, request_id: str, prompt_text: str) -> None:
        self._state[request_id] = {
            "prompt": prompt_text,
            "response": "",
            "selector_routes": [],
            "expert_routes": [],
        }

    def push_selector(self, request_id: str, delta_text: str, routed_expert) -> None:
        entry = self._state.get(request_id)
        if entry is None:
            return
        entry["response"] += delta_text
        entry["selector_routes"].append(routed_expert)

    def push_expert(self, request_id: str, expert_id, delta_text: str) -> None:
        entry = self._state.get(request_id)
        if entry is None:
            return
        entry["response"] += delta_text
        entry["expert_routes"].append(expert_id)

    def get_transcript(self, request_id: str) -> str:
        entry = self._state.get(request_id)
        if entry is None:
            return ""
        return entry["prompt"] + entry["response"]

    def finalize(self, request_id: str) -> dict | None:
        return self._state.pop(request_id, None)

