# Visualization Service Deployment

## 1. How to run
- Quick Model(DEBUG):

  ```bash
  python ./deploy/run.py
  ```

- Deployment Model(WSGI Http Server):
  > The related parameter usage method can be found in the [gunicorn documentation](https://docs.gunicorn.org/en/stable/run.html). An example is given below.

  ```bash
  gunicorn -b 0.0.0.0:8888 deploy.run:app --threads 4
  ```

## 2. How to configure
> All execution configuration parameters can be set in `deploy/dep_config.json`. This is a json file, and you need to follow the json writing specifications when editing.

### Network Field (`net`)

  - Description

|KEY|DES|DEF|
|:---|:---|:---:|
|`port`|Backend service open port|_not default_|
|`app_name`|Service access interface path name|_not default_|
|`session_time_out`|The longest life cycle (seconds) in which a session is idle|600|

  - Example

  ```json
  "net":
  {
    "port": 8787,
    "app_name": "tatk",
    "session_time_out": 300
  }
  ```


### Module Field (`nlu`, `dst`, `policy`, `nlg`)
   > The candidate models can be configured under the key values of `nlu`, `dst`, `policy`, `nlg`. The model under each module needs to set a unique name as the key.

   - Description

|KEY|DES|DEF|
|:---|:---|:---:|
|`class_path`|Target model class relative path|_not default_|
|`data_set`|The data set used by the model|_not default_|
|`ini_params`|The parameters required for the class to be instantiated|`{}`|
|`model_name`|Model name displayed on the front end|model key|
|`max_core`|The maximum number of cores this model allows to start|1|
|`preload`|If false, this model is not preloaded|`true`|
|`enable`|If false, the system will ignore this configuration|`true`|

   - Example

   ```json
   "nlu":
   {
     "svm-cam": {
      "class_path": "convlab2.nlu.svm.camrest.nlu.SVMNLU",
      "data_set": "camrest",
      "ini_params": {"mode": "usr"},
      "model_name": "svm-cam",
      "max_core": 1,
      "preload": true,
      "enable": true
    },
    "svm-mul": {
      "class_path": "convlab2.nlu.svm.multiwoz.nlu.SVMNLU",
      "data_set": "multiwoz",
      "ini_params": {"mode": "usr"},
      "model_name": "svm-mul",
      "max_core": 1,
      "preload": false,
      "enable": true
    }
   }
   ```

## 3. How to use
After starting up the service, you can open the following url in your web browser:
> http://0.0.0.0:[port]/dialog

Then you can choose an available dialog corpus and system configuration.
After that, you can start a conversation by input something in the right input area and click [send].

### Intermediate Result Modification
In this service, you can restore a dialog turn by modifying the intermediate result which you think is incorrect.
For example, if you get an incorrect DST result, you can modify its result in the input area by the left side. 

Since you modify the content, a button will appear. Then you can click the button, and the dialog system will restore this turn by masking the DST module and directly use your input as the DST result.
