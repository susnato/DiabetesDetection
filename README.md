# Usage
This trains a Xgboost model on [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The logs are stored at [logs.txt](./logs.txt)

The commands are stored at [commands.txt](./commands.txt)

Docker image of the triton api is stored at [susnato/xgb-diabetes-triton:cpu](https://hub.docker.com/repository/docker/susnato/xgb-diabetes-triton/general)
To run- 
```bash
curl -X POST http://localhost:8000/v2/models/xgb/infer -d @/home/susnato/PycharmProjects/DiabetesDetection/triton_payload.json
```
where the payload should look like - 
```json
{"inputs": [{"name": "input__0", "datatype": "FP32", "shape": [1,8], "data": [[1,85,66,29,0,26.6,0.351,31]]}]}
```

The output should be -
```bash
{"model_name":"xgb","model_version":"1","outputs":[{"name":"output__0","datatype":"FP32","shape":[1,2],"data":[0.6297283172607422,0.3702717125415802]}]}
```
