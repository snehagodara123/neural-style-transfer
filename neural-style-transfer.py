# Neural Style Transfer with TensorFlow
# This project demonstrates how to implement neural style transfer
# using TensorFlow and pre-trained VGG19 model

import os
import tensorflow as tf
import numpy as np
import PIL.Image
import time
import matplotlib.pyplot as plt
import IPython.display as display

# Function to load and preprocess images
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Function to display images
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Load the pre-trained VGG19 model
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Content layer where we'll extract content features
content_layers = ['block5_conv2'] 

# Style layers where we'll extract style features
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1', 
    'block4_conv1', 
    'block5_conv1'
]

# Number of style and content layers
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Function to calculate content loss
def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Function to calculate style loss
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def style_loss(base_style, gram_target):
    return tf.reduce_mean(tf.square(gram_matrix(base_style) - gram_target))

# Extract style and content features
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        
    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        
        content_dict = {content_name: value 
                       for content_name, value 
                       in zip(self.content_layers, content_outputs)}
        
        style_dict = {style_name: value
                     for style_name, value
                     in zip(self.style_layers, style_outputs)}
        
        return {'content': content_dict, 'style': style_dict}

# Optimizer (Adam with a custom learning rate)
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, style_targets, content_targets,
                      style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    style_loss_value = tf.add_n([style_loss(style_outputs[name], style_targets[name]) 
                                for name in style_outputs.keys()])
    style_loss_value *= style_weight / num_style_layers
    
    content_loss_value = tf.add_n([content_loss(content_outputs[name], content_targets[name])
                                  for name in content_outputs.keys()])
    content_loss_value *= content_weight / num_content_layers
    
    total_loss = style_loss_value + content_loss_value
    return total_loss

# The main function for style transfer
def run_style_transfer(content_path, style_path, epochs=10, steps_per_epoch=100):
    # Load content and style images
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    
    # Display original images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')
    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')
    plt.tight_layout()
    plt.show()
    
    # Initialize model
    extractor = StyleContentModel(style_layers, content_layers)
    
    # Extract features
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    
    # Initialize generated image with content image
    generated_image = tf.Variable(content_image)
    
    # Define hyperparameters
    style_weight = 1e-2
    content_weight = 1e4
    total_variation_weight = 30
    
    # Define optimizer
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    
    # Function to perform one step of optimization
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs, style_targets, content_targets,
                                     style_weight, content_weight)
            
            # Add total variation loss for smoother results
            loss += total_variation_weight * tf.image.total_variation(image)
            
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
        return loss
    
    # Run optimization
    start = time.time()
    step = 0
    display_interval = max(1, steps_per_epoch // 5)
    
    # Track loss
    losses = []
    
    for epoch in range(epochs):
        for i in range(steps_per_epoch):
            step += 1
            loss = train_step(generated_image)
            losses.append(loss.numpy())
            
            if step % display_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{steps_per_epoch}, Loss: {loss.numpy():.4f}")
    
    end = time.time()
    print(f"Total time: {end-start:.1f} seconds")
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Style Transfer Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()
    
    # Show final image
    plt.figure(figsize=(10, 10))
    imshow(generated_image, 'Generated Image')
    plt.show()
    
    # Save the generated image
    result_image = tensor_to_image(generated_image)
    output_path = 'stylized_image.jpg'
    result_image.save(output_path)
    print(f"Stylized image saved as {output_path}")
    
    return generated_image

# Example usage
def main():
    # Specify paths to your content and style images
    content_path = 'content_image.jpg'  # Replace with your content image path
    style_path = 'style_image.jpg'      # Replace with your style image path
    
    # Run style transfer
    generated_image = run_style_transfer(
        content_path=content_path,
        style_path=style_path,
        epochs=10,
        steps_per_epoch=100
    )
    
    # You can also try different settings
    # For example, more epochs for better results:
    # generated_image = run_style_transfer(content_path, style_path, epochs=20, steps_per_epoch=100)

if __name__ == "__main__":
    main()
