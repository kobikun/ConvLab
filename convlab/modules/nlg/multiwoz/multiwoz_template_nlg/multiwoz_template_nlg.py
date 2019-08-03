"""
template NLG for multiwoz dataset. templates are in `multiwoz_template_nlg/` dir.
See `example` function in this file for usage.
"""
import json
import os
import random
from pprint import pprint

from convlab.modules.nlg.nlg import NLG


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# supported slot
slot2word = {
    'Fee': 'fee',
    'Addr': 'address',
    'Area': 'area',
    'Stars': 'stars',
    'Internet': 'Internet',
    'Department': 'department',
    'Choice': 'choice',
    'Ref': 'reference number',
    'Food': 'food',
    'Type': 'type',
    'Price': 'price range',
    'Stay': 'stay',
    'Phone': 'phone',
    'Post': 'postcode',
    'Day': 'day',
    'Name': 'name',
    'Car': 'car type',
    'Leave': 'leave',
    'Time': 'time',
    'Arrive': 'arrive',
    'Ticket': 'ticket',
    'Depart': 'departure',
    'People': 'people',
    'Dest': 'destination',
    'Parking': 'parking',
    'Open': 'open',
    'Id': 'Id',
    # 'TrainID': 'TrainID'
}

from operator import itemgetter
def choice_jaccard(candidate_lists, intents):
    _intents = set([x.lower() for x in intents])
    _candidate_lists = [ (x, x.lower().split(" ")) for x in candidate_lists]
    _candi_score = []
    for idx, (c_sent, c_tokens) in enumerate(_candidate_lists):
        print(idx, c_sent)
        c_tokens_set = set(c_tokens)
        _union = c_tokens_set.union(_intents)
        _intersection = c_tokens_set.intersection(_intents)
        sim = 1.0*(len(_intersection)/len(_union))
        _candi_score.append(sim)
    pair_candi = list(zip([ x[0] for x in _candidate_lists], _candi_score))
    pair_candi = sorted(pair_candi, key=itemgetter(1), reverse=True)
    print(pair_candi)
    return pair_candi[0][0]





class MultiwozTemplateNLG(NLG):
    def __init__(self, is_user, mode="manual"):
        """
        :param is_user: if dialog_act from user or system
        :param mode:    `auto`: templates extracted from data without manual modification, may have no match;
                        `manual`: templates with manual modification, sometimes verbose;
                        `auto_manual`: use auto templates first. When fails, use manual templates.
        both template are dict, *_template[dialog_act][slot] is a list of templates.
        """
        super().__init__()
        self.is_user = is_user
        self.mode = mode
        template_dir = os.path.dirname(os.path.abspath(__file__))
        self.auto_user_template = read_json(os.path.join(template_dir, 'auto_user_template_nlg.json'))
        self.auto_system_template = read_json(os.path.join(template_dir, 'auto_system_template_nlg.json'))
        self.manual_user_template = read_json(os.path.join(template_dir, 'manual_user_template_nlg.json'))
        self.manual_system_template = read_json(os.path.join(template_dir, 'manual_system_template_nlg.json'))

    def generate(self, dialog_acts):
        """
        NLG for Multiwoz dataset
        :param dialog_acts: {da1:[[slot1,value1],...], da2:...}
        :return: generated sentence
        """
        mode = self.mode
        print("NLG generate\tdialog_acts:%s" % dialog_acts)
        print("NLG mode:%s" % mode)
        try:
            is_user = self.is_user
            if mode=='manual':
                if is_user:
                    template = self.manual_user_template
                else:
                    template = self.manual_system_template

                return self._manual_generate(dialog_acts, template)

            elif mode=='auto':
                if is_user:
                    template = self.auto_user_template
                else:
                    template = self.auto_system_template

                return self._auto_generate(dialog_acts, template)

            elif mode=='auto_manual':
                if is_user:
                    template1 = self.auto_user_template
                    template2 = self.manual_user_template
                else:
                    template1 = self.auto_system_template
                    template2 = self.manual_system_template

                res = self._auto_generate(dialog_acts, template1)
                if res == 'None':
                    res = self._manual_generate(dialog_acts, template2)
                return res

            else:
                raise Exception("Invalid mode! available mode: auto, manual, auto_manual")
        except Exception as e:
            print('Error in processing:')
            pprint(dialog_acts)
            raise e

    def _postprocess(self,sen):
        sen = sen.strip().capitalize()
        if len(sen) > 0 and sen[-1] != '?' and sen[-1] != '.':
            sen += '.'
        sen += ' '
        return sen

    def _manual_generate(self, dialog_acts, template):
        sentences = ''
        print("NLG manual_generate\tdialog_acts:%s" % dialog_acts)
        #print("NLG manual_generate\ttemplate:%s" % template)
        for dialog_act, slot_value_pairs in dialog_acts.items():
            print("NLG dialog-act item iter\t%s, %s" % (dialog_act, slot_value_pairs))
            intent = dialog_act.split('-')
            if 'Select'==intent[1]:
                slot2values = {}
                for slot, value in slot_value_pairs:
                    slot2values.setdefault(slot, [])
                    slot2values[slot].append(value)
                for slot, values in slot2values.items():
                    if slot == 'none': continue
                    sentence = 'Do you prefer ' + values[0]
                    for i, value in enumerate(values[1:]):
                        if i == (len(values) - 2):
                            sentence += ' or ' + value
                        else:
                            sentence += ' , ' + value
                    sentence += ' {} ? '.format(slot2word[slot])
                    sentences += sentence
            elif 'Request'==intent[1]:
                for slot, value in slot_value_pairs:
                    if dialog_act not in template or slot not in template[dialog_act]:
                        sentence = 'What is the {} of {} ? '.format(slot, dialog_act.split('-')[0].lower())
                        sentences += sentence
                    else:
                        sentence = random.choice(template[dialog_act][slot])
                        print("NLG select sentence : %s", sentence)
                        sentence = self._postprocess(sentence)
                        sentences += sentence
            elif 'general'==intent[0] and dialog_act in template:
                sentence = random.choice(template[dialog_act]['none'])
                sentence = self._postprocess(sentence)
                sentences += sentence
            else:
                for slot, value in slot_value_pairs:
                    if dialog_act in template and slot in template[dialog_act]:
                        #sentence = random.choice(template[dialog_act][slot])
                        sentence = choice_jaccard(template[dialog_act][slot], intent)

                        print("NLG select sentence : %s" % (template[dialog_act][slot]))
                        sentence = sentence.replace('#{}-{}#'.format(dialog_act.upper(), slot.upper()), str(value))
                        print("NLG select sentence : %s", sentence)
                    else:
                        if slot in slot2word:
                            _sub_ne = slot2word[slot]
                            if _sub_ne.lower() in ['phone']:
                                _sub_ne = "%s number" % (_sub_ne)
                            sentence = 'The {} {} is {} . '.format(intent[0], _sub_ne, str(value))
                            print("slot based\t", sentence)
                        else:
                            sentence = ''
                    sentence = self._postprocess(sentence)
                    sentences += sentence
        return sentences.strip()

    def _auto_generate(self, dialog_acts, template):
        sentences = ''
        for dialog_act, slot_value_pairs in dialog_acts.items():
            key = ''
            for s, v in sorted(slot_value_pairs, key=lambda x: x[0]):
                key += s + ';'
            if dialog_act in template and key in template[dialog_act]:
                sentence = random.choice(template[dialog_act][key])
                if 'Request' in dialog_act or 'general' in dialog_act:
                    sentence = self._postprocess(sentence)
                    sentences += sentence
                else:
                    for s, v in sorted(slot_value_pairs, key=lambda x: x[0]):
                        if v != 'none':
                            sentence = sentence.replace('#{}-{}#'.format(dialog_act.upper(), s.upper()), v, 1)
                    sentence = self._postprocess(sentence)
                    sentences += sentence
            else:
                return 'None'
        return sentences.strip()


def example():
    # dialog act
    dialog_acts = {}
    # whether from user or system
    is_user = False

    multiwoz_template_nlg = MultiwozTemplateNLG(is_user)
    # print(dialog_acts)
    print(multiwoz_template_nlg.generate(dialog_acts))


if __name__ == '__main__':
    example()
