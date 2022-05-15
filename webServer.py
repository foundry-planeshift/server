import json
import cv2
import asyncio
import re
import argparse

from av import VideoFrame
import aiohttp_cors
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)

from planeshift.planeShift import PlaneShift, Mode

class PlaneShiftStreamTrack(VideoStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, videostream):
        super().__init__()  # don't forget this!

        self.videostream = videostream

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(self.videostream(), format="bgr24")
        frame.pts = pts
        frame.time_base = time_base

        return frame

plane_shift = None

pcs = set()
async def original_image(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % 0
    pcs.add(pc)

    def log_info(msg, *args):
        print(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

    pc.addTrack(PlaneShiftStreamTrack(plane_shift.original_image))

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def roi_image(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % 0
    pcs.add(pc)

    def log_info(msg, *args):
        print(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

    print("Adding track")
    pc.addTrack(PlaneShiftStreamTrack(plane_shift.roi_image))

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print("Returning web response")
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def select_battlearea(_):
    ok = plane_shift.select_battlearea()

    return web.Response(
        status=200 if ok else 500
    )

def sqrt_int(X: int, N: int):
    import math
    # N = math.floor(math.sqrt(X))
    while bool(X % N):
        N -= 1
    M = X // N
    return M, N

async def calibration_image(request):
    print("Webserver: Received 'calibration_image' request.")
    json = await request.json()
    tv_width_pixels = int(json['width']) - 15
    tv_height_pixels = int(json['height'])

    tv_width_mm = 750
    pixels_to_mm = tv_width_mm/tv_width_pixels
    tv_height_mm = round(tv_height_pixels*pixels_to_mm)


    square_x, chess_size_mm = sqrt_int(tv_width_mm, 25)
    aruco_size_mm = chess_size_mm - 6

    square_y = int(tv_height_mm/chess_size_mm)

    # chess_size_mm = 25
    # aruco_size_mm = 19
    # square_x, square_y = (40, 22)

    print(f"Generating aruco board with {square_y}x{square_x} [rowxcol] and chess size {chess_size_mm}mm and aruco size {aruco_size_mm}mm")

    board = cv2.aruco.CharucoBoard_create(square_x, square_y, chess_size_mm, aruco_size_mm, cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000))
    board_image = board.draw((tv_width_pixels, tv_height_pixels))

    _, jpg = cv2.imencode('.jpg', board_image)
    return web.Response(
        status=200,
        body=jpg.tobytes(),
        content_type='image/jpeg'
    )

async def calibrate_camera(_):
    print("Webserver: Received 'calibrate_camera' request.")

    plane_shift.set_mode(Mode.CALIBRATION)
    while Mode(plane_shift._mp_mode.value) == Mode.CALIBRATION:
        # print("Waiting for calibration to finish")
        await asyncio.sleep(1)

    return web.Response(
        status=200
    )

async def tokens_location(_):
    tokens_location = plane_shift._mp_player_tokens
    location_json = json.dumps(list(tokens_location))
    return web.Response(
        status=200,
        text=location_json
    )

async def set_camera_exposure(request):
    params = await request.json()
    exposure_value = int(params['exposure'])

    print(f"Setting camera exposure to {exposure_value}")
    plane_shift.set_camera_exposure(exposure_value)

    return web.Response(
        status=200
    )

async def health_check(_):
    return web.Response(status=200)

def start_server(hostname, port):
    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
    })

    cors.add(app.router.add_post("/originalimage", original_image))
    cors.add(app.router.add_post("/roiimage", roi_image))
    cors.add(app.router.add_post("/selectroi", select_battlearea))
    cors.add(app.router.add_get("/tokenslocation", tokens_location))
    cors.add(app.router.add_post("/calibrationimage", calibration_image))
    cors.add(app.router.add_post("/calibratecamera", calibrate_camera))
    cors.add(app.router.add_post("/setcameraexposure", set_camera_exposure))
    cors.add(app.router.add_get("/healthcheck", health_check))

    web.run_app(app, host=hostname, port=port)

def parse_resolution(string):
    match = re.search("^(\d+)x(\d+)$", string)
    return match.groups()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--device', type=str, help='The video device to capture images from')
    parser.add_argument('--resolution', nargs="?", type=str, help='The port for the server')
    parser.add_argument('--hostname', nargs="?", type=str, help='The hostname for the server')
    parser.add_argument('--port', nargs="?", type=int, help='The port for the server')
    parser.add_argument('--debug', nargs="?", type=bool, help='To enable debug mode')
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        config = json.load(f)

    device = args.device if (args.device) else config.get('device').get('location')
    if device is None:
        print("No device defined in configuration. Exitting.")
        exit()

    resolution = parse_resolution(args.resolution) if (args.resolution) else parse_resolution(config.get('device').get('resolution'))
    if len(resolution) != 2:
        print("No resolution defined in configuration. Exitting.")
        exit()

    hostname = args.hostname if (args.hostname) else config.get('server').get('hostname')
    if hostname is None:
        hostname = "localhost"

    port = args.port if (args.port) else config.get('server').get('port')
    if port is None:
        port = 8337

    debug = args.debug if (args.debug) else config.get('debug')
    if debug is None:
        debug = False

    print(f"Starting PlaneShift server with the following settings:\n"
          f"Device: {device}\n"
          f"Resolution: {resolution[0]}x{resolution[1]}\n"
          f"Hostname: {hostname}\n"
          f"Port: {port}\n"
          f"Debugging: {debug}")

    plane_shift = PlaneShift(device, resolution, debug=debug)
    plane_shift.load_camera_calibration("calibration/calibration.json")
    plane_shift.start()

    start_server(hostname, port)