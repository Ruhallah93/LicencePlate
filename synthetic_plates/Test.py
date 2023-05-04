from threading import Thread

threadList = []


def generator(decision_time):
    models = ['GRU']
    # 'GRU','LSTM','ConvLSTM','BiLSTM'
    # 'Power_consumption','PRSA2017','RSSI','User_Identification_From_Walking','WISDM','Motor_Failure_Time'
    for problem in ['WISDM']:
        dataset = problems[problem]['dataset']
        n_classes = problems[problem]['n_classes']
        features = problems[problem]['features']
        sample_rate = problems[problem]['sample_rate']
        data_length_time = problems[problem]['data_length_time']
        n_h_block = problems[problem]['n_h_block']
        n_train_h_block = problems[problem]['n_train_h_block']
        n_valid_h_block = problems[problem]['n_valid_h_block']
        n_test_h_block = problems[problem]['n_test_h_block']
        h_moving_step = problems[problem]['h_moving_step']
        segments_times = problems[problem]['segments_times']

        log_dir = f"./comparisons/log/{problem}/recurrent/"

        for model in models:
            for decision_overlap in [0.0]:
                for segments_time in segments_times:
                    for segments_overlap in [0.75]:
                        if float(decision_time * 0.75) < float(segments_time):
                            continue

                        if (os.path.exists(log_dir + "statistics.csv")):
                            is_investigated = False
                            saved_statistics = pd.read_csv(log_dir + "statistics.csv")
                            for index, row in saved_statistics.iterrows():
                                if (row['segments_time'] == str(timedelta(seconds=int(segments_time)))
                                        and row['decision_time'] == str(timedelta(seconds=int(decision_time)))):
                                    print("dl: " + row['decision_time'] + " and wl:" + row[
                                        'segments_time'] + " is investigated!")
                                    is_investigated = True
                                    break
                            if (is_investigated):
                                continue

                        print("dl: " + str(timedelta(seconds=int(decision_time))) +
                              " and " +
                              "wl: " + str(timedelta(seconds=int(segments_time))))

                        classifier = eval(model)(classes=n_classes,
                                                 n_features=len(features),
                                                 segments_size=int(segments_time * sample_rate),
                                                 segments_overlap=segments_overlap,
                                                 decision_size=int(decision_time * sample_rate),
                                                 decision_overlap=decision_overlap)

                        # cross-validation
                        statistics = h_block_analyzer(db_path=dataset,
                                                      sample_rate=sample_rate,
                                                      features=features,
                                                      n_classes=n_classes,
                                                      noise_rate=noise_rate,
                                                      segments_time=segments_time,
                                                      segments_overlap=segments_overlap,
                                                      decision_time=decision_time,
                                                      decision_overlap=decision_overlap,
                                                      classifier=classifier,
                                                      epochs=epochs,
                                                      batch_size=batch_size,
                                                      data_length_time=data_length_time,
                                                      n_h_block=n_h_block,
                                                      n_train_h_block=n_train_h_block,
                                                      n_valid_h_block=n_valid_h_block,
                                                      n_test_h_block=n_test_h_block,
                                                      h_moving_step=h_moving_step)


decision_times = [10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60, 6 * 60, 7 * 60, 8 * 60, 9 * 60, 10 * 60]
for i in range(len(decision_times)):
    print("Tread ", i + 1, " is running")
    t = Thread(target=generator,
               args=(decision_times[i]))
    t.start()
    threadList.append(t)
for t in threadList:
    t.join()
