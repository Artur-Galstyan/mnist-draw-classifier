import asyncio
import tornado

from mnist_draw_classifier.model import CNN
import jax
import jax.numpy as jnp
import equinox as eqx
import logging
import json
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_model():
    key = jax.random.PRNGKey(5678)
    model = CNN(key)
    model = eqx.tree_deserialise_leaves("mnist.eqx", model)
    return model


model = get_model()


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

    def post(self):
        LOGGER.info("Received request")
        body_json = json.loads(self.request.body)
        data = jnp.array(body_json["data"], dtype=jnp.float32).reshape(28, 28)
        data = jnp.rot90(data, -1)
        data = jnp.fliplr(data)
        data = data.reshape(1, 28, 28)
        plt.imsave("test.png", data[0], cmap="gray")

        pred_y = model(data)

        LOGGER.info("Prediction: %s", pred_y)
        best_guess = jnp.argmax(pred_y).item()
        second_best_guess = jnp.argsort(pred_y)[-2].item()
        self.write({"best_guess": best_guess, "second_best_guess": second_best_guess})


class CopyHandler(tornado.web.RequestHandler):
    def post(self):
        LOGGER.info("Received request")
        body_json = json.loads(self.request.body)
        data = jnp.array(body_json["data"], dtype=jnp.float32).reshape(28, 28)
        data = jnp.rot90(data, -1)
        data = jnp.fliplr(data)
        rows = cols = len(data)
        gpt_string = "You are given a 28x28 grid of pixels, similar to MNIST, where -1 means black and 1 means white. Guess the number given the following pixels: "  # noqa
        for row in range(rows):
            for col in range(cols):
                gpt_string += f"x={row} y={col} = {data[row][col]}\n"

        gpt_string += "The number is: "
        self.write({"gpt_string": gpt_string})


def make_app():
    return tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/copy", CopyHandler),
        ],
        debug=True,
    )


async def main():
    app = make_app()
    app.listen(8888)
    await asyncio.Event().wait()


if __name__ == "__main__":
    LOGGER.info("Starting server")
    asyncio.run(main())
