import grpc
import numpy as np
from src.generated import inference_pb2, inference_pb2_grpc
from src.triton_client import TritonClient, triton_to_numpy_dtype, numpy_to_triton_dtype


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, triton_client: TritonClient):
        self._triton = triton_client

    def Infer(self, request, context):
        try:
            inputs_data, datatype_map = _parse_input_tensors(request.inputs)
            output_names = [o.name for o in request.outputs]

            results = self._triton.infer(
                model_name=request.model_name,
                inputs_data=inputs_data,
                output_names=output_names,
                model_version=request.model_version,
                datatype_override=datatype_map,
            )

            response = inference_pb2.InferResponse(
                model_name=request.model_name,
                model_version=request.model_version,
            )
            for name, arr in results.items():
                response.outputs.append(
                    inference_pb2.InferOutputTensor(
                        name=name,
                        datatype=numpy_to_triton_dtype(arr.dtype),
                        shape=list(arr.shape),
                        raw_data=arr.tobytes(),
                    )
                )
            return response

        except Exception as e:
            print(f"[InferenceServicer] Infer 오류: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.InferResponse(error=str(e))

    def InferStream(self, request_iterator, context):
        """양방향 스트리밍: 클라이언트 요청마다 즉시 응답 반환."""
        for request in request_iterator:
            if context.is_active():
                yield self.Infer(request, context)

    def ServerHealth(self, request, context):
        return inference_pb2.HealthResponse(
            live=self._triton.is_alive(),
            ready=self._triton.is_ready(),
        )

    def ModelReady(self, request, context):
        ready = self._triton.is_model_ready(
            request.model_name, request.model_version
        )
        return inference_pb2.ModelReadyResponse(
            model_name=request.model_name,
            ready=ready,
        )


def _parse_input_tensors(
    tensors,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    inputs_data: dict[str, np.ndarray] = {}
    datatype_map: dict[str, str] = {}

    for tensor in tensors:
        dtype = triton_to_numpy_dtype(tensor.datatype)
        arr = np.frombuffer(tensor.raw_data, dtype=dtype)
        if tensor.shape:
            arr = arr.reshape(list(tensor.shape))
        inputs_data[tensor.name] = arr
        datatype_map[tensor.name] = tensor.datatype

    return inputs_data, datatype_map
