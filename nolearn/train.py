
def train_step(self,  mini_batches, num_batches, verbose=1):        
        """Train a mini-batch --> single epoch"""

        # set timer for epoch run
        performance = MonitorPerformance(verbose)
        performance.set_start_time(start_time = time.time())

        # train on mini-batch with random shuffling
        for index in range(num_batches):
            X, y = next(mini_batches)
            cost, prediction = self.train_fun(X, y)
            performance.add_cost(cost)
            performance.progress_bar(index, num_batches)
        print "" 
        return performance.get_mean_cost()


#!/bin/python
import sys
from neuralnetwork import MonitorPerformance
from utils import batch_generator
from six.moves import cPickle


def train_minibatch(nnmodel, train, valid, batch_size=128, num_epochs=500, 
            patience=10, verbose=1, filepath='.'):
    """Train a model with cross-validation data and test data"""

    # setup generator for mini-batches
    num_train_batches = len(train[0]) // batch_size
    train_batches = batch_generator(train[0], train[1], batch_size)

    num_valid_batches = len(valid[0]) // batch_size
    valid_batches = batch_generator(valid[0], valid[1], batch_size)

    # train model
    for epoch in range(num_epochs):
        if verbose == 1:
            sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

        # training set
        train_cost = nnmodel.train_step(train_batches, num_train_batches, verbose)
        nnmodel.train_monitor.add_cost(train_cost)

        # test current model with cross-validation data and store results
        valid_cost, valid_prediction = nnmodel.test_step_batch(valid)
        nnmodel.valid_monitor.update(valid_cost, valid_prediction, valid[1])
        nnmodel.valid_monitor.print_results("valid")
        
        # save model
        if filepath:
            savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
            nnmodel.save_model_parameters(savepath)

        # check for early stopping                  
        status = nnmodel.valid_monitor.early_stopping(valid_cost, epoch, patience)
        if not status:
            break

    return nnmodel



def train_valid_minibatch(nnmodel, train, valid, batch_size=128, num_epochs=500, 
            patience=10, verbose=1, filepath='.'):
    """Train a model with cross-validation data and test data"""

    # setup generator for mini-batches
    num_train_batches = len(train[0]) // batch_size
    train_batches = batch_generator(train[0], train[1], batch_size)

    num_valid_batches = len(valid[0]) // batch_size
    valid_batches = batch_generator(valid[0], valid[1], batch_size)

    # train model
    for epoch in range(num_epochs):
        if verbose == 1:
            sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

        # training set
        train_cost = nnmodel.train_step(train_batches, num_train_batches, verbose)
        nnmodel.train_monitor.add_cost(train_cost)

        # test current model with cross-validation data and store results
        valid_cost = nnmodel.test_step_minibatch(train_batches, num_train_batches)
        nnmodel.valid_monitor.add_cost(valid_cost)
        nnmodel.valid_monitor.print_results("valid")
        
        # save model
        if filepath:
            savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
            nnmodel.save_model_parameters(savepath)

        # check for early stopping                  
        status = nnmodel.valid_monitor.early_stopping(valid_cost, epoch, patience)
        if not status:
            break

    return nnmodel
    

def test_model_all(nnmodel, test, num_train_epochs, filepath):
    """loops through training parameters for epochs min_index 
    to max_index located in filepath and calculates metrics for 
    test data """
    print "Model performance for each training epoch on on test data set"

    performance = MonitorPerformance("test_all")
    for epoch in range(num_train_epochs):
        sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_train_epochs))

        # build a new neural network
        nnmodel.reinitialize()

        # load model parameters for a given training epoch
        savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
        f = open(savepath, 'rb')
        best_parameters = cPickle.load(f)
        f.close()

        # get test metrics 
        nnmodel.set_model_parameters(best_parameters)
        test_cost, test_prediction = nnmodel.test_step_batch(test)
        performance.update(test_cost, test_prediction, test[1])
        performance.print_results("test") 

    return performance



#------------------------------------------------------------------------------------

def train_loop(self, X, y, epochs=None):
        epochs = epochs or self.max_epochs
        X_train, X_valid, y_train, y_valid = self.train_split(X, y, self)

        on_batch_finished = self.on_batch_finished
        if not isinstance(on_batch_finished, (list, tuple)):
            on_batch_finished = [on_batch_finished]

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_started = self.on_training_started
        if not isinstance(on_training_started, (list, tuple)):
            on_training_started = [on_training_started]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        for func in on_training_started:
            func(self, self.train_history_)

        num_epochs_past = len(self.train_history_)

        while epoch < epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            if self.custom_scores:
                custom_scores = [[] for _ in self.custom_scores]
            else:
                custom_scores = []

            t0 = time()

            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.apply_batch_func(
                    self.train_iter_, Xb, yb)
                train_losses.append(batch_train_loss)

                for func in on_batch_finished:
                    func(self, self.train_history_)

            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.apply_batch_func(
                    self.eval_iter_, Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)

                if self.custom_scores:
                    y_prob = self.apply_batch_func(self.predict_iter_, Xb)
                    for custom_scorer, custom_score in zip(self.custom_scores, custom_scores):
                        custom_score.append(custom_scorer[1](yb, y_prob))

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            if custom_scores:
                avg_custom_scores = np.mean(custom_scores, axis=1)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss == avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss == avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'dur': time() - t0,
                }
            if self.custom_scores:
                for index, custom_score in enumerate(self.custom_scores):
                    info[custom_score[0]] = avg_custom_scores[index]
            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)