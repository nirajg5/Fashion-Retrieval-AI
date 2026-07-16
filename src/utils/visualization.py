"""
Visualization Utilities
"""

import matplotlib.pyplot as plt

from PIL import Image


class Visualizer:

    def show(self, image_path):

        image = Image.open(image_path)

        plt.figure(figsize=(5,5))

        plt.imshow(image)

        plt.axis("off")

        plt.show()

    def show_grid(self, results):

        n = len(results)

        plt.figure(figsize=(15,10))

        for i, result in enumerate(results):

            plt.subplot(

                (n+2)//3,

                3,

                i+1

            )

            image = Image.open(

                result["image_path"]

            )

            plt.imshow(image)

            plt.title(

                f"{result['final_score']:.3f}"

            )

            plt.axis("off")

        plt.tight_layout()

        plt.show()


visualizer = Visualizer()