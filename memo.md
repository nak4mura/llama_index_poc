## llama.cppへのプロンプトとレスポンスをprintするMOD
- venv/lib/pythonX.X/site-packages/llama_index/llms/llama_cpp/base.py
``` python
    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # MOD START
        print("\n=== Prompt ===\n")
        print(prompt)
        print("\n==============\n")
        # MOD END
        self.generate_kwargs.update({"stream": False})

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        response = self._model(prompt=prompt, **self.generate_kwargs)
        # MOD START
        print("\n=== Response ===\n")
        print(response["choices"][0]["text"])
        print("\n==============\n")
        # MOD END
        return CompletionResponse(text=response["choices"][0]["text"], raw=response)
```