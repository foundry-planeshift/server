
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

but webserver requires some parameters to be able to run. There are two ways to provide this information:
1. Using a configuration file (see [config.json](config.json) for an example)
```json
{
  "device": {  
    "location": "/dev/video1", // Location of the video device (webcam)
    "resolution": "2048x1536" // Resolution to run the webcam on
  },
  "server": {
    "hostname": "localhost", // Location of the webserver, used to connect the module with
    "port": 8337 // Port of the webserver, used to connect the module with
  },
  "debug": true // Enable debug mode
}
```
2. Providing the parameters using the CLI
```commandline
python webServer.py [--device /dev/video1] [--resolution 2048x1536] [--hostname localhost] [--port 8337] [--debug] [--config config.json]
```

## Module
The FoundryVTT module itself.
See [the module page](https://github.com/foundry-planeshift/module/tree/main).
