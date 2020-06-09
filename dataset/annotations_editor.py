import cv2
import argparse
import os
from utils.screenshot_processor import ScreenshotProcessor
from ocr.data.image_processing import letter_boxes
from ocr.data.visualization import draw_boxes
from utils.path import abs_path
import numpy as np
from utils.path import make_dir_if_needed


class Annotator(ScreenshotProcessor):

    def __init__(self, input_dir, output_dir):
        super().__init__(input_dir)
        self.output_dir = output_dir
        self.letter_boxes = []
        self.selected_box_index = None
        self.add_event('d', self._delete_selected_box_action)
        self.add_event('s', self._save_annotation)
        self.set_mouse_callback(self.handle_click)

    def _delete_selected_box_action(self, image, name):
        if self.selected_box_index is not None:
            self.letter_boxes.pop(self.selected_box_index)
            self.reset()
            self.current_screenshot_copy = draw_boxes(self.current_screenshot_copy, self.letter_boxes)
            self.selected_box_index = None

    def _save_annotation(self, image, name):
        boxes = np.array(self.letter_boxes)
        np.save(os.path.join(self.output_dir, name), boxes)
        print('Saving success')

    def preprocess_image(self, screenshot):
        self.letter_boxes = letter_boxes(screenshot).tolist()
        return draw_boxes(screenshot, self.letter_boxes)

    def handle_click(self, screenshot, event, point):
        if event == cv2.EVENT_LBUTTONDOWN:
            def point_in_box(p, b):
                px, py = p
                bx, by, w, h = b
                return bx <= px <= (bx + w) and by <= py <= (by + h)
            self.selected_box_index = next(iter([i for i, box in enumerate(self.letter_boxes) if point_in_box(point, box)]),
                                           None)
            print('Selected box index:', self.selected_box_index)


def main(args):
    make_dir_if_needed(args.output_dir)
    annotator = Annotator(args.input_dir, args.output_dir)
    annotator.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=abs_path('data/images/raw'))
    parser.add_argument('--output_dir', default=abs_path('data/annotations/'))
    main(parser.parse_args())

