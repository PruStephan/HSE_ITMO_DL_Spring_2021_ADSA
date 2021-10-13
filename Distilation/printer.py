import matplotlib.pyplot as plt


def read_logs(model_name):
    file_path = 'logs/{0}.log'.format(model_name)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return list(map(lambda line: float(line.strip()), lines))


def main():
    with_teacher_accs = read_logs('with_teacher')
    teacher_accs = read_logs('teacher')
    without_teacher_accs = read_logs('without_teacher')

    plt.rcParams["figure.figsize"] = (20, 10)
    plt.plot([i for i in range(len(with_teacher_accs))], with_teacher_accs, label='with_teacher')
    plt.plot([i for i in range(len(teacher_accs))], teacher_accs, label='teacher')
    plt.plot([i for i in range(len(without_teacher_accs))], without_teacher_accs, label='without_teacher')
    plt.legend(loc=4, prop={'size': 15})
    plt.show()


if __name__ == '__main__':
    main()
