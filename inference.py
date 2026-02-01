import os, re, cv2
from typing import Union
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class UnifiedInference:
    """
    A unified class for performing inference using Thinker models.
    Supports 4B (non-thinking) models.
    """
    
    def __init__(self, model_id="UBTECH-Robotics/Thinker-4B", device_map="auto"):
        """
        Initialize the model and processor.
        
        Args:
            model_id (str): Path or Hugging Face model identifier
            device_map (str): Device mapping strategy ("auto", "cuda:0", etc.)
        """
        print("Loading Checkpoint ...")
        self.model_id = model_id
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, 
                dtype="auto", 
                device_map="auto"
                )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def inference(self, text: str, image: Union[list, str], task="general", 
                 plot=False, do_sample=True, temperature=0.7):
        """
        Perform inference with text and images input.
        
        Args:
            text (str): The input text prompt.
            image (Union[list,str]): The input image(s) as a list of file paths or a single file path.
            task (str): The task type, e.g., "general", "pointing", "affordance", "trajectory", "grounding".
            plot (bool): Whether to plot results on image.
            enable_thinking (bool, optional): Whether to enable thinking mode. 
                                            If None, auto-determined based on model capability.
            do_sample (bool): Whether to use sampling during generation.
            temperature (float): Temperature for sampling.
        """

        if isinstance(image, str):
            image = [image]

        assert task in ["general", "pointing", "grounding"], \
            f"Invalid task type: {task}. Supported tasks are 'general', 'pointing', 'grounding'."
        assert task == "general" or (task in ["pointing", "grounding"] and len(image) == 1), \
            "Pointing and grounding tasks require exactly one image."

        if task == "pointing":
            print("Pointing task detected. Adding pointing prompt.")
            text = f"{text}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."
        elif task == "grounding":
            print("Grounding task detected. Adding grounding prompt.")
            text = f"{text}. Please generate a set of bounding box (bbox) coordinates based on the image and description.The bbox coordinate format is [top-left x, top-left y, bottom-right x, bottom-right y].All values must be integer points between 0 and 1000, inclusive."

        print(f"\n{'='*20} INPUT {'='*20}\n{text}\n{'='*47}\n")

        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", 
                         "image": path 
                        } for path in image
                    ],
                    {"type": "text", "text": f"{text}"},
                ],
            },
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer_text = output_text[0] if output_text else ""

        # Plotting functionality
        if plot and task in ["pointing","grounding"]:
            print("Plotting enabled. Drawing results on the image ...")

            plot_points, plot_boxes, plot_trajectories = None, None, None
            result_text = answer_text  # Use the processed answer text for plotting

            if task == "pointing":
                point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                points = re.findall(point_pattern, result_text)
                plot_points = [(int(x), int(y)) for x, y in points]
                print(f"Extracted points: {plot_points}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_pointing_annotated.")
            elif task == "grounding":
                box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
                boxes = re.findall(box_pattern, result_text)
                plot_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]
                print(f"Extracted bounding boxes: {plot_boxes}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_grounding_annotated.")

            os.makedirs("result", exist_ok=True)
            image_path_to_save = os.path.join("result", image_name_to_save)

            self.draw_on_image(
                image[0],
                points=plot_points,
                boxes=plot_boxes,
                output_path=image_path_to_save
            )

        # Return unified format
        result = {"answer": answer_text}
        
        return result

    def draw_on_image(self, image_path, points=None, boxes=None,output_path=None):
        """
        Draw points, bounding boxes, and trajectories on an image

        Parameters:
            image_path: Path to the input image
            points: List of points in format [(x1, y1), (x2, y2), ...]
            boxes: List of boxes in format [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
            trajectories: List of trajectories in format [[(x1, y1), (x2, y2), ...], [...]]
            output_path: Path to save the output image. Default adds "_annotated" suffix to input path
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")
            img_h, img_w = image.shape[:2]

            # Draw points
            if points:
                for (rel_x, rel_y) in points:
                    norm_x = rel_x / 1000.0
                    norm_y = rel_y / 1000.0
                    abs_x = norm_x * img_w
                    abs_y = norm_y * img_h
                    x, y = int(round(abs_x)), int(round(abs_y))
                    cv2.circle(image, (x, y), 8, (0, 0, 255), 4)

            # Draw bounding boxes
            if boxes:
                for box_item in boxes:
                    if isinstance(box_item, dict):
                        bbox_rel = box_item["bbox_2d"]
                        label = box_item.get("label", "")
                    elif isinstance(box_item, list) and len(box_item) == 4:
                        bbox_rel = box_item
                        label = ""
                    else:
                        print(f"invalid box format: {box_item}")
                        continue

                    rel_x1, rel_y1, rel_x2, rel_y2 = bbox_rel
                    norm_x1 = rel_x1 / 1000.0
                    norm_y1 = rel_y1 / 1000.0
                    norm_x2 = rel_x2 / 1000.0
                    norm_y2 = rel_y2 / 1000.0
                    abs_x1 = norm_x1 * img_w
                    abs_y1 = norm_y1 * img_h
                    abs_x2 = norm_x2 * img_w
                    abs_y2 = norm_y2 * img_h
                    x1, y1 = int(round(abs_x1)), int(round(abs_y1))
                    x2, y2 = int(round(abs_x2)), int(round(abs_y2))
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)

                    if label:
                        text_x = max(x1 + 5, 0)
                        text_y = max(y1 - 5, 20)
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6, 2)
                        cv2.rectangle(image, (text_x, text_y - text_h - 5),(text_x + text_w, text_y + 5), (0, 0, 255), -1)
                        cv2.putText(image, label, (text_x, text_y),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 2)

            # Determine output path
            if not output_path:
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_annotated{ext}"

            # Save the result
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error processing image: {e}")
            return None


# Usage examples
if __name__ == "__main__":
    
    print("=== Testing Thinker-4B Model ===")
    model_path = 'UBTECH-Robotics/Thinker-4B'
    model_4b = UnifiedInference(model_path)

    # ============general task==========
    prompt = "What is shown in this image?"
    image = "http://images.cocodataset.org/val2017/000000039769.jpg"
    predition = model_4b.inference(prompt, image, task="general", plot=True)
    print(f"Prediction:\n{predition}")

    # ============2d-point grounding task==========
    prompt = "Please point out the free space between two cats."
    image = "http://images.cocodataset.org/val2017/000000039769.jpg"
    predition = model_4b.inference(prompt, image, task="pointing", plot=True)
    print(f"Prediction:\n{predition}")

    # ============2d-box grounding task==========
    prompt = "Please detect the cat on the right."
    image = "http://images.cocodataset.org/val2017/000000039769.jpg"
    predition = model_4b.inference(prompt, image, task="grounding", plot=True)
    print(f"Prediction:\n{predition}")

