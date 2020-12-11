import torch.utils.data
import torch.optim as optim
from skimage.draw import circle_perimeter
from skimage.io import imsave
from tqdm import tqdm
import copy
import os
from config import Config
import centroidnet
from centroidnet import CentroidNetV2, CentroidLossV2, fit_circle
import numpy as np


def create_centroidnet(num_channels, num_classes):
    model = CentroidNetV2(num_classes, num_channels)
    return model


def create_centroidnet_loss():
    loss = CentroidLossV2()
    return loss


def validate(validation_loss, epoch, validation_set_loader, model, loss, validation_interval=10):
    if epoch % validation_interval == 0:
        with torch.no_grad():
            # Validate using validation data loader
            model.eval()  # put in evaluation mode
            validation_loss = 0
            idx = 0
            for inputs, targets in validation_set_loader:
                inputs = inputs.to(Config.dev)
                targets = targets.to(Config.dev)
                outputs = model(inputs)
                mse = loss(outputs, targets)
                validation_loss += mse.item()
                idx += 1
            model.train()  # put back in training mode
            return validation_loss / idx
    else:
        return validation_loss


def save_model(filename, model):
    print(f"Save snapshot to: {os.path.abspath(filename)}")
    with open(filename, "wb") as f:
        torch.save(model.state_dict(), f)


def load_model(filename, model):
    print(f"Load snapshot from: {os.path.abspath(filename)}")
    with open(filename, "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    return model


def train(training_set, validation_set, model, loss, epochs, batch_size, learn_rate, validation_interval):
    print(f"Training {len(training_set)} images for {epochs} epochs with a batch size of {batch_size}.\n"
          f"Validate {len(validation_set)} images each {validation_interval} epochs and learning rate {learn_rate}.\n")

    best_model = copy.deepcopy(model)
    model.to(Config.dev)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                                      shuffle=True, num_workers=10, drop_last=True)
    validation_set_loader = torch.utils.data.DataLoader(validation_set, batch_size=min(len(validation_set), batch_size),
                                                        shuffle=True, num_workers=10, drop_last=True)

    if len(training_set_loader) == 0:
        raise Exception("The training dataset does no contain any samples. "
                        "Is the minibatch larger than the amount of samples?")
    if len(training_set_loader) == 0:
        raise Exception("The validation dataset does no contain any samples. "
                        "Is the minibatch larger than the amount of samples?")

    bar = tqdm(range(1, epochs))
    validation_loss = 9999
    best_loss = 9999
    for epoch in bar:
        training_loss = 0
        idx = 0
        # Train one mini batch
        for (inputs, targets) in training_set_loader:
            inputs = inputs.to(Config.dev)
            targets = targets.to(Config.dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            ls = loss(outputs, targets)
            ls.backward()
            optimizer.step()
            training_loss += ls.item()
            idx += 1

        scheduler.step(epoch)

        # Update progress bar
        bar.set_description("Epoch {}/{} Loss(T): {:5f} and Loss(V): {:.5f}"
                            .format(epoch, epochs, training_loss / idx, validation_loss))
        bar.refresh()

        # Validate and save
        validation_loss = validate(validation_loss, epoch, validation_set_loader, model, loss, validation_interval)
        if validation_loss < best_loss:
            print(f"Update model with loss {validation_loss:5f}")
            best_loss = validation_loss
            best_model.load_state_dict(model.state_dict())

    return best_model


def predict(data_set, model, loss, max_dist, nm_window, centroid_threshold, border_threshold):
    print(f"Predicting {len(data_set)} files with loss {type(loss)}")
    with torch.no_grad():
        data_set.eval()
        model.eval()
        model.to(Config.dev)
        loss_value = 0
        idx = 0
        set_loader = torch.utils.data.DataLoader(data_set, batch_size=5, shuffle=False, num_workers=1, drop_last=False)
        result_images = []
        result_objects = []
        for inputs, targets in tqdm(set_loader):
            inputs = inputs.to(Config.dev)
            targets = targets.to(Config.dev)
            outputs = model(inputs)
            ls = loss(outputs, targets)
            loss_value += ls.item()
            decoded = [centroidnet.decode(img, max_dist, nm_window, centroid_threshold, border_threshold)
                       for img in outputs.cpu().numpy()]

            # Add all numpy arrays to a list (disable this to save memory)
            result_images.extend([{"inputs": i.cpu().numpy(),
                                   "targets": t.cpu().numpy(),
                                   "class_ids": d[2],
                                   "class_probs": d[3],
                                   "centroid_vectors": d[4],
                                   "border_vectors": d[5],
                                   "centroid_votes": d[6],
                                   "border_votes": d[7]} for i, t, o, d in zip(inputs, targets, outputs, decoded)])

            # Use the centroid and the border points to fit a circle (many shapes could be fitted)
            for results_per_image in decoded:
                objects_per_image = []
                for centroid, borders in zip(results_per_image[0], results_per_image[1]):
                    if borders.shape[1] < 1:
                        break
                    x, y, r = fit_circle(centroid, borders)
                    class_id = results_per_image[2][0, y, x]
                    class_prob = results_per_image[3][0, y, x]
                    objects_per_image.append([x, y, r, class_id, class_prob])
                result_objects.append(objects_per_image)

            idx = idx + 1
    print("Aggregated loss is {:.5f}".format(loss_value / idx))
    return result_images, result_objects


def output(folder, result_images, result_objects):
    os.makedirs(folder, exist_ok=True)
    print(f"Created output folder {os.path.abspath(folder)}")
    for i, sample in enumerate(result_images):
        for name, arr in sample.items():
            np.save(os.path.join(folder, f"{i}_{name}.npy"), arr)

    for i, (objects, images) in enumerate(zip(result_objects, result_images)):
        image = images["inputs"]
        image = ((image * Config.div) + Config.sub).astype(np.uint8)
        image = np.moveaxis(image, 0, 2)
        for object in objects:
            x, y, r, class_id, _ = object
            rr, cc = circle_perimeter(y, x, r)

            if 0 < class_id <= 3:
                image[rr, cc, class_id - 1] = 255
            else:
                image[rr, cc, :] = 255  # else make the circle white.

        imsave(os.path.join(folder, f"{i}_overlay.png"), image)

    lines = ["image_nr x y r class_id class_prob \r\n"]
    with open(os.path.join(folder, "validation.txt"), "w") as f:
        for i, image in enumerate(result_objects):
            for line in image:
                line_str = [str(i), *[str(elm) for elm in line]]
                lines.append(" ".join(line_str) + "\r\n")
        f.writelines(lines)


def main(do_train=True, do_predict=True):
    # Load validation dataset
    validation_set = centroidnet.CentroidNetDataset(os.path.join("data", "dataset", "validation.csv"),
                                                    crop=None, max_dist=Config.max_dist,
                                                    sub=Config.sub, div=Config.div)
    # Create loss function
    loss = create_centroidnet_loss()

    if do_train:
        # Load training dataset
        training_set = centroidnet.CentroidNetDataset(os.path.join("data", "dataset", "training.csv"),
                                                      crop=Config.crop, max_dist=Config.max_dist,
                                                      sub=Config.sub, div=Config.div)

        assert training_set.num_classes == Config.num_classes, f"Number of classes on config.py is incorrect. " \
                                                               f"Should be {training_set.num_classes}"

        # Create network and load snapshots
        model = create_centroidnet(num_channels=Config.num_channels, num_classes=Config.num_classes)
        model = train(training_set, validation_set, model, loss,
                      epochs=Config.epochs, batch_size=Config.batch_size, learn_rate=Config.learn_rate,
                      validation_interval=Config.validation_interval)
        save_model(os.path.join("data", "CentroidNet.pth"), model)

    # Load model
    model = create_centroidnet(num_channels=3, num_classes=Config.num_classes)
    model = load_model(os.path.join("data", "CentroidNet.pth"), model)

    # Predict
    if do_predict:
        result_images, result_centroids = predict(validation_set, model, loss,
                                                  max_dist=Config.max_dist,
                                                  nm_window=Config.nm_size,
                                                  centroid_threshold=Config.centroid_threshold,
                                                  border_threshold=Config.border_threshold)
        output(os.path.join("data", "validation_result"), result_images, result_centroids)


if __name__ == '__main__':
    main(do_train=False)
