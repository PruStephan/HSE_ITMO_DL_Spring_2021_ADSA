import model as m
import torch


def load_saved_model(model_name, layers_count):
    model = m.ResNet(n=layers_count)
    teacher_weights = torch.load('models/{0}.pt'.format(model_name))
    model.load_state_dict(teacher_weights)
    return model


def main(name, layers=2, epochs=100, use_teacher=False, teacher_name=None, teacher_layers=None):
    test_dataset = m.download_dataset(train=False, save_path='data')

    train_dataset = m.download_dataset(train=True, save_path='data')

    model = m.ResNet(n=layers)
    epochs_passed = 0

    if not use_teacher:
        accs = m.train_model(model=model, epochs=epochs,
                             train_dataset=train_dataset, test_dataset=test_dataset,
                             epochs_passed=epochs_passed)
    else:
        teacher_model = load_saved_model(model_name=teacher_name, layers_count=teacher_layers)
        accs = m.train_model(model=model, epochs=epochs,
                             teacher=teacher_model, alpha=0.5,
                             train_dataset=train_dataset, test_dataset=test_dataset,
                             epochs_passed=epochs_passed)
    torch.save(model.state_dict(), 'models/{0}.pt'.format(name))
    with open('logs/{0}.log'.format(name), 'w') as f:
        for acc in accs:
            f.write('{0}\n'.format(acc))

    acc = m.accuracy(model=model, test_dataset=test_dataset)
    print('Accuracy = {0}'.format(acc))


if __name__ == '__main__':
    main('without_teacher')
