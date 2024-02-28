from functools import lru_cache

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_classifier(input: str) -> dict:
    triton_client = get_client()

    model_input = InferInput(name="float_input", shape=input.shape, datatype="FP32")
    model_input.set_data_from_numpy(input, binary_data=True)

    query_response = triton_client.infer(
        "titanic_classifier",
        [model_input],
        outputs=[InferRequestedOutput("probabilities", binary_data=True)],
    )

    dict_response = {"probabilities": query_response.as_numpy("probabilities")}
    return dict_response


def main():
    input = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=np.float32,
    )

    output = call_triton_classifier(input)
    print(f"Probabilities:\n{output['probabilities']}")


if __name__ == "__main__":
    main()
