import os
import torchvision
from PIL import Image
import torch
import pickle
from .DeepAutoma import ProbabilisticAutoma
from .NN_models import CNN_grounder, Linear_grounder_no_droput

from statistics import mean
from sklearn.model_selection import train_test_split

from .utils import eval_acceptance, eval_learnt_DFA_acceptance, eval_image_classification_from_traces
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
import time

def create_batches_same_length(dataset, labels, size):
    new_dataset = []
    new_labels = []
    num_batches = int(len(dataset)/size)
    for i in range(num_batches):
        batch_trace = []
        batch_label = []
        for j in range(size):
            batch_trace.append(dataset[i*size+j])
            batch_label.append(labels[i*size+j])
        batch_trace = torch.stack(batch_trace)
        batch_label = torch.stack(batch_label)
        new_dataset.append(batch_trace)
        new_labels.append(batch_label)
    return new_dataset, new_labels 

class NeuralRewardMachine:
    def __init__(self, numb_states, numb_symbols, numb_rewards, num_exp=0,log_dir="Results/", dataset="minecraft_location"):
        self.first_training = False
        self.ltl_formula_string = "goal"
        self.log_dir = log_dir
        self.exp_num=num_exp

        self.numb_of_symbols = numb_symbols
        self.numb_of_states = numb_states
        self.numb_of_rewards = numb_rewards

        self.alphabet = ["c"+str(i) for i in range(self.numb_of_symbols) ]

        #################### networks
        self.hidden_dim =numb_states

        ##### DeepDFA
        self.deepAutoma = ProbabilisticAutoma(self.numb_of_symbols, self.numb_of_states, self.numb_of_rewards)

        ##### Classifier
        self.dataset = dataset
        if dataset == 'minecraft_image':
            self.num_classes = 5
            self.num_channels = 3

            self.pixels_h = 64
            self.pixels_v = 64

            self.num_features = 4 #<---??
            self.classifier = CNN_grounder(self.num_classes)

        if dataset == 'minecraft_location':
            self.num_inputs = 2
            self.num_classes = 5

            self.classifier = Linear_grounder_no_droput(self.num_inputs, 128, self.num_classes)

        self.temperature = 1.0
        #questa resize si può togliere mi sà
        resize = torchvision.transforms.Resize((64,64))
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            resize,
        ])

        #queste cose sotto misà che servivano per fare la image classification
        '''
        if dataset == 'minecraft_image':
            trace = []
            dir = os.listdir('custom_trace_whole')
            for i in range(len(dir)):
                img = Image.open('custom_trace_whole/img'+str(i)+'.jpg')
                img = transforms(img)
                trace.append(img)
            self.custom_trace = [torch.stack(trace).unsqueeze(0)]

            trace = []
        '''
        if dataset == 'minecraft_location':
            self.custom_trace = [torch.tensor([[0,0],[0,1],[0,2],[0,3],
                                             [1,0],[1,1],[1,2],[1,3],
                                             [2,0],[2,1],[2,2],[2,3],
                                             [3,0],[3,1],[3,2],[3,3]]).unsqueeze(0)]

        self.symbolic_grid = [torch.tensor([[0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,0,1,0],
                                    [0,0,0,0,1], [1,0,0,0,0], [0,0,0,0,1], [0,0,0,0,1],
                                    [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1],
                                    [0,0,1,0,0], [0,0,0,0,1], [0,0,0,0,1], [0,1,0,0,0]])]
        #[0,0,0,0,1] white cell
        #[0,0,0,1,0] gem
        #[0,0,1,0,0] door
        #[0,1,0,0,0] lava
        #[1,0,0,0,0] pick

    # def set_dataset(self, image_traj, rew_traj):

    #     # print(len(rew_traj[0]))

    #     dataset_traces = []
    #     dataset_acceptances = torch.FloatTensor(rew_traj)
    #     for i in range(len(image_traj)):
    #         trace = []
    #         for img in image_traj[i]:
    #             trace.append(img)
    #         trace_tensor = torch.stack(trace)
    #         trace_tensor = torch.squeeze(trace_tensor)
    #         dataset_traces.append(trace_tensor)

    #     #train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = train_test_split(dataset_traces, dataset_acceptances, train_size=1, shuffle=True)
    #     train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = (dataset_traces, dataset_traces, dataset_acceptances, dataset_acceptances)

    #     train_img_seq, train_acceptance_img = create_batches_same_length(train_traces, train_acceptance_tr, 40)


    #     test_img_seq_hard, test_acceptance_img_hard = train_img_seq, train_acceptance_img

    #     image_seq_dataset = (train_img_seq, [], train_acceptance_img, test_img_seq_hard, [], test_acceptance_img_hard)
    #     self.train_img_seq, self.train_traces, self.train_acceptance_img, self.test_img_seq_hard, self.test_traces, self.test_acceptance_img_hard = image_seq_dataset
    #     return

    def set_dataset(self, image_traj, rew_traj, lengths=None):

        dataset_acceptances = torch.LongTensor(rew_traj)
        # image_traj puo' essere gia' un tensore (traiettorie, passi, ...) oppure una lista di liste di tensori
        dataset_traces = image_traj if torch.is_tensor(image_traj) else torch.stack([torch.stack(inner) for inner in image_traj])
        dataset_lengths = torch.LongTensor(lengths) if lengths is not None else None

        image_seq_dataset = ([dataset_traces], [], [dataset_acceptances], [dataset_traces], [], [dataset_acceptances])
        self.train_img_seq, self.train_traces, self.train_acceptance_img, self.test_img_seq_hard, self.test_traces, self.test_acceptance_img_hard = image_seq_dataset
        self.train_lengths = [dataset_lengths] if dataset_lengths is not None else None
        self.test_lengths_hard = self.train_lengths

        return image_seq_dataset

    def eval_learnt_DFA(self, automa_implementation, temp, mode="dev"):
        if mode=="dev":
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.dev_traces, self.dev_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr), automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.dev_traces, self.dev_acceptance_tr), automa_implementation, temp)
        else:
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp)
        return train_acc, test_acc


    def train_symbol_grounding(self, num_of_epochs):

        #print(self.deepAutoma.trans_prob)
        #print(self.deepAutoma.rew_matrix)
        #assert False

        tot_size = len(self.train_img_seq)

        self.deepAutoma.to(device)
        '''
        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_NS_exp"+str(self.exp_num), 'w')
        test_hard_file = open(self.log_dir+self.ltl_formula_string+"_test_hard_acc_NS_exp"+str(self.exp_num), 'w')
        image_classification_train_file = open(self.log_dir+self.ltl_formula_string+"_image_classification_train_acc_NS_exp"+str(self.exp_num), 'w')
        image_classification_test_file = open(self.log_dir+self.ltl_formula_string+"_image_classification_test_acc_NS_exp"+str(self.exp_num), 'w')
        '''
        self.classifier.to(device)

        class_weights = getattr(self, 'class_weights', None)
        if class_weights is not None:
            cross_entr = torch.nn.CrossEntropyLoss(reduction='none', weight=class_weights.to(device))
        else:
            cross_entr = torch.nn.CrossEntropyLoss(reduction='none')
        print("_____________training the GROUNDER_____________")
        print("training on {} sequences using {} automaton states".format(tot_size, self.numb_of_states))

        if not hasattr(self, 'optimizer'):
            params = self.classifier.parameters()
            if self.dataset == "minecraft_location":
                self.optimizer = torch.optim.Adam(params, lr=0.001)#, weight_decay=1e-3)
            else:
                self.optimizer = torch.optim.Adam(params, lr=0.001)#, weight_decay=1e-3)

            self.sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, min_lr=1e-05)

        epoch = 0
        mean_loss = 1000000
        max_accuracy = 0
        best_classifier = self.classifier

        for _ in range(num_of_epochs):
        #while True:
            #if epoch % 40 == 0:
            print("epoch: ", epoch)
            epoch+=1
            losses = []

            self.classifier.train()
            for b in range(len(self.train_img_seq)):

                batch_img_seq = self.train_img_seq[b].to(device)

                batch_size = batch_img_seq.size()[0]
                length_seq = batch_img_seq.size()[1]
                target_rew_seq = self.train_acceptance_img[b].type(torch.int64).to(device)

                self.optimizer.zero_grad()
                if self.dataset == 'minecraft_image':
                    flat_input = batch_img_seq.view(-1, self.num_channels, self.pixels_v, self.pixels_h)
                else:
                    flat_input = batch_img_seq.float().view(-1, self.num_inputs)

                lengths_b = self.train_lengths[b] if getattr(self, 'train_lengths', None) is not None else None
                if lengths_b is not None:
                    lengths_b = lengths_b.to(device)
                    # il classifier vede solo gli step reali: gli step di padding (copie dell'ultimo
                    # step) altrimenti entrano nelle statistiche della BatchNorm e le sbilanciano
                    # verso le caselle finali. Ai passi di padding si ricopiano poi i simboli
                    # dell'ultimo step reale, che e' la stessa osservazione.
                    real = (torch.arange(length_seq, device=device).unsqueeze(0) < lengths_b.unsqueeze(1))
                    sym_flat = torch.zeros(batch_size * length_seq, self.numb_of_symbols, device=device)
                    sym_flat[real.reshape(-1)] = self.classifier(flat_input[real.reshape(-1)])
                    last_real = torch.minimum(torch.arange(length_seq, device=device).unsqueeze(0), (lengths_b - 1).unsqueeze(1))
                    sym_sequences = sym_flat.view(batch_size, length_seq, self.numb_of_symbols).gather(
                        1, last_real.unsqueeze(-1).expand(-1, -1, self.numb_of_symbols))
                else:
                    sym_sequences = self.classifier(flat_input).view(batch_size, length_seq, self.numb_of_symbols)

                pred_states, pred_rew = self.deepAutoma(sym_sequences, self.temperature)

                pred_rew = pred_rew.view(-1, self.numb_of_rewards).to(device)
                target_rew = target_rew_seq.view(-1)
                if self.deepAutoma.normalized:
                    # pred_rew e' gia' una distribuzione: log-verosimiglianza della classe vera
                    per_position_loss = torch.nn.functional.nll_loss(
                        torch.log(pred_rew.clamp_min(1e-8)), target_rew, reduction='none',
                        weight=class_weights.to(device) if class_weights is not None else None)
                else:
                    per_position_loss = cross_entr(pred_rew, target_rew)

                if lengths_b is not None:
                    # esclude gli step di padding (ripetizione dell'ultimo step reale) dalla loss,
                    # cosi' il training si concentra sulla dinamica vera invece che su una
                    # predizione gia' facile/ripetuta
                    mask = (torch.arange(length_seq, device=device).unsqueeze(0) < lengths_b.unsqueeze(1)).reshape(-1)
                    loss = (per_position_loss * mask).sum() / mask.sum()
                else:
                    loss = per_position_loss.mean()

                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
            
            
            mean_loss_new = mean(losses)
            # niente sheduler.step() qui: questa funzione viene chiamata una volta per batch/chunk
            # (non una volta per epoch), farlo qui sovra-campiona lo scheduler di ~10x rispetto
            # all'epoch reale e causa un collasso prematuro del learning rate. Lo step va fatto
            # una sola volta per epoch, in A2C_nrm.py, sulla loss media dell'intero epoch.

            if mean_loss_new < 0.3 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            if epoch > num_of_epochs/2 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            mean_loss = mean_loss_new

            train_accuracy, test_accuracy_hard = self.eval_all(automa_implementation='logic_circuit', temperature=1, discretize_labels=True)
            if train_accuracy>= max_accuracy:
                max_accuracy = train_accuracy
                best_classifier = self.classifier

            if self.dataset == "minecraft_location":
                train_image_classification_accuracy, test_image_classification_accuracy = self.eval_image_classification()
            else:
                train_image_classification_accuracy = 0
                test_image_classification_accuracy = 0
            
            #if epoch % 40 == 0:
            print("__________________________")
            print("MEAN LOSS: ", mean_loss_new)
            print("SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy : {}".format(train_accuracy, test_accuracy_hard))
            print("IMAGE CLASSIFICATION: train accuracy : {}\ttest accuracy : {}".format(train_image_classification_accuracy,test_image_classification_accuracy))

            mean_loss = mean_loss_new
        #write the accuracies of the last epoch

        self.classifier = best_classifier    
        return mean_loss


    def train_DFA(self, batch_size, num_of_epochs, decay=0.999, freezed=False):
        def get_lr(optim):
            for param_group in optim.param_groups:
                return param_group['lr']

        tot_size = len(self.train_traces)
        mean_loss = 1000000

        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_NS_exp"+str(self.exp_num), 'w')
        dev_file = open(self.log_dir+self.ltl_formula_string+"_dev_acc_NS_exp"+str(self.exp_num), 'w')

        train_file_dfa = open(self.log_dir+self.ltl_formula_string+"_train_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        dev_file_dfa = open(self.log_dir+self.ltl_formula_string+"_dev_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        test_file_dfa = open(self.log_dir+self.ltl_formula_string+"_test_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        loss_file = open(self.log_dir+self.ltl_formula_string+"_loss_dfa_NS_exp"+str(self.exp_num), 'w')

        cross_entr = torch.nn.CrossEntropyLoss()
        print("_____________training the DFA_____________")
        print("training on {} sequences using {} automaton states".format(tot_size, self.numb_of_states))

        params = [self.deepAutoma.trans_prob] + [self.deepAutoma.rew_matrix]
        optimizer = torch.optim.Adam(params, lr=0.01)
        sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-04)


        min_temp = 0.00001
        self.temperature =1.0

        if freezed:
            self.temperature = min_temp

        start_time = time.time()
        epoch= -1
        while True:
            epoch+=1
            print("epoch: ", epoch)
            losses = []
            for i in range(len(self.train_traces)):

                batch_trace_dataset = self.train_traces[i].to(device)
                batch_acceptance = self.train_acceptance_tr[i].to(device)
                optimizer.zero_grad()

                predictions= self.deepAutoma(batch_trace_dataset, self.temperature)

                loss = cross_entr(predictions, batch_acceptance)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=self.temperature)
            mean_loss_new = mean(losses)
            print("SEQUENCE CLASSIFICATION (LOGIC CIRCUIT): train accuracy : {}\ttest accuracy : {}\tloss : {}".format(train_accuracy, test_accuracy, mean_loss_new))

            train_file.write("{}\n".format(train_accuracy))
            dev_file.write("{}\n".format(test_accuracy))
            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=min_temp)
            print("SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy : {}".format(train_accuracy, test_accuracy))

            train_file_dfa.write("{}\n".format(train_accuracy))
            dev_file_dfa.write("{}\n".format(test_accuracy))
            loss_file.write("{}\n".format(mean(losses)))
            if freezed:
                self.temperature = min_temp
            else:
                self.temperature = max(self.temperature*decay, min_temp)
            print("temp: ", self.temperature)

            sheduler.step(mean_loss_new)
            print("lr: ", get_lr(optimizer))
            if mean_loss_new < 0.318 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            if epoch > 200 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            mean_loss = mean_loss_new


        ######################## net2dfa
        #save the minimized dfa
        self.dfa = self.deepAutoma.net2dfa( min_temp)
        ex_time =  time.time() - start_time

        with open("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+".ex_time", "w") as f:
            f.write("{}\n".format(ex_time))

        #print it
        try:
            self.dfa.to_graphviz().render("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_minimized.dot")
        except:
            print("Not able to render automa")
        with open("DFA_predicted_nesy/"+self.ltl_formula_string, 'wb') as outp:
            pickle.dump(self.dfa, outp, pickle.HIGHEST_PROTOCOL)

        with open("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_min_num_states", "w") as f:
            f.write(str(len(self.dfa._states)))

        #ULTIMO TEST usando il DFA sul TEST set
        train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='dfa', temp=min_temp, mode="test")
        print("FINAL SEQUENCE CLASSIFICATION ON TEST SET: {}".format(test_accuracy))

        test_file_dfa.write("{}\n".format(test_accuracy))

    def eval_counts(self, automa_implementation, temperature, discretize_labels=False):
        # conteggi (corretti/totali, anche per classe) sul dataset impostato con set_dataset
        return eval_acceptance(self.classifier, self.deepAutoma, self.alphabet, (self.train_img_seq, self.train_acceptance_img), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True, lengths=getattr(self, 'train_lengths', None), return_counts=True)

    def eval_all(self, automa_implementation, temperature, discretize_labels=False, return_balanced=False):
        # eval_acceptance esclude gia' il padding dal conteggio quando ha le lengths
        if return_balanced:
            train_accuracy, balanced_accuracy = eval_acceptance(self.classifier, self.deepAutoma, self.alphabet, (self.train_img_seq, self.train_acceptance_img), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True, lengths=getattr(self, 'train_lengths', None), return_balanced=True)
        else:
            train_accuracy = eval_acceptance(self.classifier, self.deepAutoma, self.alphabet, (self.train_img_seq, self.train_acceptance_img), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True, lengths=getattr(self, 'train_lengths', None))

        test_accuracy_hard = eval_acceptance( self.classifier, self.deepAutoma, self.alphabet,(self.test_img_seq_hard, self.test_acceptance_img_hard), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True, lengths=getattr(self, 'test_lengths_hard', None))

        if return_balanced:
            return train_accuracy, test_accuracy_hard, balanced_accuracy
        return train_accuracy, test_accuracy_hard

    def eval_image_classification(self):
        train_acc = eval_image_classification_from_traces(self.custom_trace, self.symbolic_grid, self.classifier, True)
        test_acc = train_acc
        return train_acc, test_acc
