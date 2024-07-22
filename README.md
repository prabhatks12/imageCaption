# Image captioning

## Setup

1. Get the dataset from Kaggle, https://www.kaggle.com/datasets/adityajn105/flickr8k 
2. Create a directory `archive/` 
3. Move the dataset to that directory

Install mlflow locally using pip.

```
pip install mlflow==2.12.1
```

To run the server, use the command:
```
mlflow server --host 127.0.0.1 --port 8080
```

## Tensorflow for Mac Arm

If using mac os with m1 or m2 chip, follow the instructions to get ternsorflow https://developer.apple.com/metal/tensorflow-plugin/


## Serving the model

1. If using version, use the command (env-manager is to use local env)
```
mlflow models serve -m "models:/<model_name>/<version>" --port 5002 --env-manager=local
```

2. If using the alias, use the @ followed by alias
```
mlflow models serve -m "models:/<model_name>@<aliases>" --port 5002 --env-manager=local
```

3. If using the run id use
```
mlflow models serve -m runs:/<run_id>/<model_name> --host 0.0.0.0 --port 5001
```

## Installing java for spark

1. For mac, insall xcode first, might take around 10 mins
```
xcode-select --install 
```

2. Install java
```
brew install java
```

3. export the path:
```
echo 'export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"' >> ~/.zshrc
export CPPFLAGS="-I/opt/homebrew/opt/openjdk/include"' 
export JAVA_HOME=`/usr/libexec/java_home`
```

4. Still getting the error while checking for java -version
```
sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk \         
     /Library/Java/JavaVirtualMachines/openjdk.jdk
```

## Extra 

1. Killing the task, ports get occupied repeatedly 
```
lsof -i tcp:8080
kill -p <pid>
```

2. Additional package used

```
pip install Flask plotly pandas
```