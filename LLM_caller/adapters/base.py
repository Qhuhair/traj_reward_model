class BaseAdapter:
    def invoke(self, prompt: str, image_before: str = None,
               image_after: str = None) -> str:
        raise NotImplementedError