def half_output(likelihood, transition):    # optional output checker for likelihood and transition probability
    likelihood_out = open("output/likelihood_out.txt", "w")
    transition_out = open("output/transition_out.txt", "w")

    for k,v in likelihood.items():
        likelihood_out.write(str(k)+" >>> "+str(v)+"\n"+"\n")

    for k,v in transition.items():
        transition_out.write(str(k)+" >>> "+str(v)+"\n"+"\n")

    likelihood_out.close()
    transition_out.close()


def trainHMM():
    in_file = open("data/training_development.pos", "r")

    likelihood = {}
    transition = {}

    # frequencies
    prev_line = "\n"
    
    # set of words
    words = set()

    # improved OOV handler
    # idea:
    # to have a dictionay storing tag-distinct prob for OOV words
    # unknown_words[tag] = number of appearances of "tag" in OOV words /  number of appearances of "tag" in training corpus
    UNKNOWN_WORD = {}

    for curr_line in in_file:
        try:
            word = curr_line.rstrip("\n").split()[0]
            curr_tag = curr_line.rstrip("\n").split()[1]
            
        except:
            # likelihood & transition (curr_tag error)
            word = "Begin_Sent"
            curr_tag= "Begin_Sent"
            likelihood[curr_tag] = likelihood.get(curr_tag, {})
            likelihood[curr_tag][word] = likelihood[curr_tag].get(word, 0) + 1
            words.add(word)
            word = "End_Sent"
            curr_tag = "End_Sent"

        try:
            prev_tag = prev_line.rstrip("\n").split()[1]

        except:
            # transition (prev_tag error)
            prev_tag = "Begin_Sent"

        # normal scenario
        # likelihood
        likelihood[curr_tag] = likelihood.get(curr_tag, {})
        likelihood[curr_tag][word] = likelihood[curr_tag].get(word, 0) + 1

        # transition
        transition[prev_tag] = transition.get(prev_tag, {})
        transition[prev_tag][curr_tag] = transition[prev_tag].get(curr_tag, 0) + 1

        prev_line = curr_line

        # set of words
        words.add(word)
        
    in_file.close()

    # dictionary for the total number of all the separate tags
    TAG_TOTAL = {}

    # probabilities
    for tag in likelihood:
        total = 0
        for aprs in likelihood[tag].values():
            total += aprs
        TAG_TOTAL[tag] = total
        for word, aprs in likelihood[tag].items():
            likelihood[tag][word] = aprs/total

            # make a list of words in training file that
            # only appears once
            if aprs == 1:
                UNKNOWN_WORD[tag] = UNKNOWN_WORD.get(tag, 0) + 1
    
    for tag in UNKNOWN_WORD:
        UNKNOWN_WORD[tag] *= 1 / TAG_TOTAL[tag]

    # print(UNKNOWN_WORD)

    # print(likelihood)

    for tag in transition:
        total = 0
        for aprs in transition[tag].values():
            total += aprs
        for word, aprs in transition[tag].items():
            transition[tag][word] = aprs/total
    # print(transition)
    
    # output
    # half_output(likelihood, transition)

    # print(transition["JJR"]["VBG"])

    # viterbi
    viterbi(likelihood, transition, words, UNKNOWN_WORD)


def viterbi(likelihood, transition, words, UNKNOWN_WORD):
    in_file = open("data/test.words", "r")
    # test_out_file = open("output/test_tags.txt", "w")
    out_file = open("output/submission.pos", "w")

    field = {}
    back_pointer = {}
    idx = 0
    tags = []

    for word in in_file:
        word = word.rstrip("\n")
        
        # print(word)

        idx += 1

        # record tag with maximum likelihood
        # and previous tag with viterbi_prob
        maxTag=""
        maxLikelihood=0.0
        maxPrevTag=""
        maxPrevLikelihood=0.0
        zeroPath = True

        if word != "":
            for tag, word_prob in likelihood.items():    # compute word likelihood prob                
                if idx == 1:    # if first word in sentence, viterbi_prob = 1*tran_prob*likelihood_prob
                    field[idx] = field.get(idx, {})

                    # handle OOV
                    if word not in words:
                        if (tag!= "Begin_Sent") & (tag!="End_Sent"):
                            try:
                                word_prob[word] = UNKNOWN_WORD[tag]    # improved OOV handler
                            except KeyError:
                                word_prob[word] = word_prob.get(word, 1/1000)    # default OOV handler
                
                    field[idx][tag] = transition["Begin_Sent"].get(tag, 0) * word_prob.get(word, 0)
                    back_pointer[idx] = back_pointer.get(idx, {})
                    back_pointer[idx][tag] = "Begin_Sent"
                else:    # if other words in sentence, viterbi_prob = prev_viterbi_prob*tran_prob*likelihood_prob
                    field[idx] = field.get(idx, {})
                    field[idx][tag] = 0    # initialize viterbi_prob for word under its current tag
                    back_pointer[idx] = back_pointer.get(idx, {})
                    
                    # handle OOV
                    if word not in words:
                        if (tag!= "Begin_Sent") & (tag!="End_Sent"):
                            try:
                                word_prob[word] = UNKNOWN_WORD[tag]    # improved OOV handler
                            except KeyError:
                                word_prob[word] = word_prob.get(word, 1/1000)    # default OOV handler
                    
                    # word is not OOV, and newly added OOV word
                    tmp_viterbi_prob = word_prob.get(word, 0)    # fetch likelihood prob for word under its current tag
                    if tmp_viterbi_prob>maxLikelihood:
                        maxTag=tag
                        maxLikelihood = tmp_viterbi_prob

                    # print(idx, tag, word, tmp_viterbi_prob)

                    for prev_tag, prev_prob in field[idx-1].items():
                        
                        if prev_prob != 0:
                            if prev_prob > maxPrevLikelihood:
                                maxPrevTag = prev_tag
                                maxPrevLikelihood = prev_prob
                            
                            tmp_viterbi_prob = word_prob.get(word, 0)
                            tmp_viterbi_prob *= transition[prev_tag].get(tag, 0)
                           
                            # print(idx,prev_tag,tag,tmp_viterbi_prob)
                            # print(idx,prev_tag,tag,field[idx-1][prev_tag],tmp_viterbi_prob)
                            tmp_viterbi_prob *= 10 # this is a no-choice workaround to deal with Python's limited capability to handle really, really small floating numbers
                            
                            tmp_viterbi_prob *= field[idx-1][prev_tag]
                            
                            # print(word,idx,prev_tag,tag,tmp_viterbi_prob)
                            if tmp_viterbi_prob > field[idx][tag]:    # update field[idx][tag] to get max viterbi_prob
                                field[idx][tag] = tmp_viterbi_prob
                                back_pointer[idx][tag] = prev_tag    # point back to prev_tag for output
                                zeroPath =False
                                # print(back_pointer[idx])
                                # print("*"*60)
            if zeroPath:    # manually set field and backpointer for zeroPath word
                field[idx][maxTag]=maxLikelihood
                back_pointer[idx][maxTag]=maxPrevTag
        else:
            field[idx] = field.get(idx, {})
            field[idx]["End_Sent"] = 0    # initialize viterbi_prob for word under its current tag
            back_pointer[idx] = back_pointer.get(idx, {})
            tmp_viterbi_prob = 1    # fetch likelihood prob for word under its current tag
            for prev_tag, prev_prob in field[idx-1].items():
                if prev_prob != 0:
                    tmp_viterbi_prob *= transition[prev_tag].get("End_Sent", 0)
                    tmp_viterbi_prob *= field[idx-1][prev_tag]
                    if tmp_viterbi_prob > field[idx]["End_Sent"]:    # update field[idx]["End_Sent"] to get max viterbi_prob
                        field[idx]["End_Sent"] = tmp_viterbi_prob
                        back_pointer[idx]["End_Sent"] = prev_tag    # point back to prev_tag for output
            # print(idx,"|||",back_pointer)
            
            # print("="*100)

            # find finalized tags for words in sentence
            # print(back_pointer)
            states = ["End_Sent"]
            state = "End_Sent"
            for n in range(idx, 0, -1):
                # print(idx,state)
                state = back_pointer[n][state]
                # print("here")
                states.append(state)
                
            states.reverse()

            # print(states)

            # test output the tags
            # for tag in states:
            #     test_out_file.write(tag+"\n")

            # prepare a list storing valid tags for final output
            for tag in states:
                if tag == "Begin_Sent":
                    pass
                else:
                    if tag == "End_Sent":
                        tag = "\n"
                    tags.append(tag+"\n")

            field = {}
            back_pointer = {}
            idx = 0
    
    # print(tags)

    in_file.close()

    in_file = open("data/test.words", "r")

    # add the words to output file
    count = 0
    for line in in_file:
        if line != "\n":
            line = "\t".join([line.rstrip("\n"), tags[count]])
        out_file.write(line)
        count += 1

    in_file.close()
    # test_out_file.close()
    out_file.close()


trainHMM()