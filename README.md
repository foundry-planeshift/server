
# PlaneShift module

PlaneShift is a [FoundryVTT](https://foundryvtt.com/) module that allows [ArUco markers](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) 
to be tracked within Foundry when using an horizontally mounted TV. This allows for a simple and cheap way to play with physical tokens on a digital system.

The system uses printed ArUco markers and a webcam for the tracking.
The software is split into two components:
## Server
The server is where the computer vision processing is done.

### Setup

To install the dev environment run:
```
python -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### Run

Running the server can be done by executing the `webServer.py` file:
```commandline
python webServer.py
```

but webserver requires some parameters to be able to run:
```
Device location - The location of the camera file, e.g. /dev/video
Device resolution - The resolution the camera should run at, e.g. 2048x1536

Server hostname - The hostname where to reach the server, e.g. localhost
Server port - The port where to reach the server, e.g. 8337

Debug - Whether to run the server in debug mode
```
There are two ways to provide this information:
1. Using a configuration file (see [config.json](config.json) for an example)
```json
{
  "device": {  
    "location": "/dev/video1",
    "resolution": "2048x1536"
  },
  "server": {
    "hostname": "localhost",
    "port": 8337
  },
  "debug": true
}
```
2. Providing the parameters using the CLI
```commandline
python webServer.py [--device /dev/video1] [--resolution 2048x1536] [--hostname localhost] [--port 8337] [--debug] [--config config.json]
```

The parameters can be given in a mix of config file and CLI, but the CLI parameters will always supersede the config file.

The server will default to certain values if those are not provided:

| Parameter | Default     |
|-----------|-------------|
| Hostname  | localhost   |
| Port      | 8337        |
| Config    | config.json |
| Debug     | false       |

## Module
The FoundryVTT module itself.
See [the module page](https://github.com/foundry-planeshift/module/tree/main).
