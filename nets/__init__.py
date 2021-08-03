# from .srcnn import *
# from .vdsr import *
# from .srgan import *
# from .esrgan import *

# from registry import MODEL_REGISTRY, CALLBACK_REGISTRY

# def build_gan(opts):
#     callbacks = opts.get("callbacks")
#     compiled_callbacks = []
#     for method, items in callbacks.items():
#         callback = CALLBACK_REGISTRY.get(method)
#         compiled_callbacks.append(callback(items, opts))

#     build_net = MODEL_REGISTRY.get(opts.get("net"))

#     batch_size = opts.get("batch_size")
#     epochs = opts.get("epochs")

#     image_manager = ImagesManager(opts)

#     image_sequence = ImageSequence(image_manager, 1)
#     image_manager.initialize_dirs(2, epochs)

#     net = build_net(opts)

#     try:
#         net.load_weights(opts.get("checkpoint_path"))
#     except Exception:
#         pass

#     history = net.fit(
#         image_sequence,
#         batch_size=batch_size,
#         epochs=epochs,
#         #  use_multiprocessing=True,
#         #  workers=2,
#         callbacks=compiled_callbacks,
#     )

# def build_cnn(opts):
#     callbacks = opts.get("callbacks")
#     compiled_callbacks = []
#     for method, items in callbacks.items():
#         callback = CALLBACK_REGISTRY.get(method)
#         compiled_callbacks.append(callback(items, opts))

#     build_net = MODEL_REGISTRY.get(opts.get("net"))

#     batch_size = opts.get("batch_size")
#     epochs = opts.get("epochs")

#     image_manager = ImagesManager(opts)

#     image_sequence = ImageSequence(image_manager, 1)
#     image_manager.initialize_dirs(2, epochs)

#     net = build_net(opts)

#     try:
#         net.load_weights(opts.get("checkpoint_path"))
#     except Exception:
#         pass

#     history = net.fit(
#         image_sequence,
#         batch_size=batch_size,
#         epochs=epochs,
#         #  use_multiprocessing=True,
#         #  workers=2,
#         callbacks=compiled_callbacks,