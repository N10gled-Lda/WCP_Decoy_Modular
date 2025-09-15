from unittest import TestCase
import unittest
import random
import sys

sys.path.append("../..")

from ErrorReconciliation.cascade.key import Key
from ErrorReconciliation.cascade.shuffle import Shuffle
from ErrorReconciliation.cascade.reconciliation import Reconciliation
from ErrorReconciliation.cascade.mock_classical_channel import MockClassicalChannel
from ErrorReconciliation.cascade.block import Block



class TestReconciliation(TestCase):

    def setUp(self, seed=2, key_size=100):
        Key.set_random_seed(seed)
        Shuffle.set_random_seed(seed + 1)

        #error_rate = round(random.uniform(0.0, 0.11),4)
        error_rate = 0.08

        #Test to perform considering that the estimation is not correct
        #estimated_bit_error_rate = round(random.uniform(0.0, 0.11),4)
        #estimated_bit_error_rate = error_rate
        estimated_bit_error_rate = 0.11

        #print(f"ERROR RATE: {error_rate}")
        #print(f"ESTIMATED BIT ERROR RATE: {estimated_bit_error_rate}")

        self.correct_key = Key.create_random_key(key_size)
        self.noisy_key = self.correct_key.copy(error_rate, Key.ERROR_METHOD_EXACT)

        print(f"Noisy Key:\n{self.noisy_key}")
        print(f"Correct Key:\n{self.correct_key}")

        self.mock_classical_channel = MockClassicalChannel(self.correct_key)

        self.reconciliation = Reconciliation("original", self.mock_classical_channel, self.noisy_key, estimated_bit_error_rate)


    def test_reconciliation(self):
        reconciled_key = self.reconciliation.reconcile()

        remaining_bit_errors = self.correct_key.difference(reconciled_key)
        print(f"Remaining_bit_errors: {remaining_bit_errors}")

        self.assertEqual(str(self.correct_key), str(reconciled_key))

    def test_ask_correct_parity_first_iteration(self):

        blocks = []

        shuffle = Shuffle(self.noisy_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)

        block_1 = Block(self.noisy_key, shuffle, 0, 15, None)
        blocks.append(block_1)
        block_2 = Block(self.noisy_key, shuffle, 15, 30, None)
        blocks.append(block_2)
        block_3 = Block(self.noisy_key, shuffle, 30, 45, None)
        blocks.append(block_3)
        block_4 = Block(self.noisy_key, shuffle, 45, 60, None)
        blocks.append(block_4)
        block_5 = Block(self.noisy_key, shuffle, 60, 75, None)
        blocks.append(block_5)
        block_6 = Block(self.noisy_key, shuffle, 75, 90, None)
        blocks.append(block_6)
        block_7 = Block(self.noisy_key, shuffle, 90, 100, None)
        blocks.append(block_7)

        for block in blocks:
            self.reconciliation._schedule_ask_correct_parity(block, False)

        self.reconciliation._service_pending_ask_correct_parity_secure_optimized()

        self.assertEqual(block_1.get_correct_parity(), 0)
        self.assertEqual(block_2.get_correct_parity(), 0)
        self.assertEqual(block_3.get_correct_parity(), 1)
        self.assertEqual(block_4.get_correct_parity(), 1)
        self.assertEqual(block_5.get_correct_parity(), 1)
        self.assertEqual(block_6.get_correct_parity(), 0)
        self.assertEqual(block_7.get_correct_parity(), 1)


    def test_ask_correct_parity_w_create_blocks(self):
        shuffle = Shuffle(self.noisy_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)

        block_size = 15

        blocks = Block.create_covering_blocks(self.noisy_key, shuffle, 15)

        for block in blocks:
            self.reconciliation._schedule_ask_correct_parity(block, False)

        print(f"Number of Blocks: {len(blocks)}")
        self.reconciliation._service_pending_ask_correct_parity_secure_optimized()

        correct_parity = [0,0,1,1,1,0,1]

        for block, parity in zip(blocks, correct_parity):
            self.assertEqual(block.get_correct_parity(), parity)


    #This test the case which the correct parity of a block is unknown and Parent Block is None making it impossible to infer the correct parity
    def test_correct_parity_is_known_or_can_be_inferred_get_correct_parity_not_asking(self):

        shuffle = Shuffle(self.noisy_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)

        block_size = 15

        blocks = Block.create_covering_blocks(self.noisy_key, shuffle, block_size)

        for block in blocks:
            self.assertIsNone(block.get_correct_parity())
            self.assertIsNone(block.get_parent_block())
            self.assertFalse(self.reconciliation._correct_parity_is_known_or_can_be_inferred(block))


    #This test the case where after asking for the parities the correct parity can be inferred since it was asked and is known
    def test_correct_parity_is_known_or_can_be_inferred_get_correct_parity_asking_not_parent_block(self):
        shuffle = Shuffle(self.noisy_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)

        block_size = 15

        blocks = Block.create_covering_blocks(self.noisy_key, shuffle, block_size)

        for block in blocks:
            self.reconciliation._schedule_ask_correct_parity(block, False)

        self.reconciliation._service_pending_ask_correct_parity_secure_optimized()

        for block in blocks:
            self.assertIsNotNone(block.get_correct_parity())
            self.assertTrue(self.reconciliation._correct_parity_is_known_or_can_be_inferred(block))

    def test_correct_parity_is_known_or_can_be_inferred_get_correct_parity_asking_has_parent_block_no_right_sibling(self):
        shuffle = Shuffle(self.noisy_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)

        block_size = 15

        blocks = Block.create_covering_blocks(self.noisy_key, shuffle, block_size)

        for block in blocks:
            self.reconciliation._schedule_ask_correct_parity(block, False)

        self.reconciliation._service_pending_ask_correct_parity_secure_optimized()

        left_sub_blocks = []

        for block in blocks:
            left_sub_block = block.create_left_sub_block()
            left_sub_blocks.append(left_sub_block)

        # Cannot infer if there is no sibling block (yet).
        for left_block in left_sub_blocks[:-1]:
            self.assertIsNone(left_block.get_correct_parity())
            self.assertIsNotNone(left_block.get_parent_block())
            self.assertIsNone(left_block.get_parent_block().get_right_sub_block())
            self.assertEqual(left_block.get_end_index() - left_block.get_start_index(), 8)
            self.assertFalse(self.reconciliation._correct_parity_is_known_or_can_be_inferred(left_block))


        #Final SubBlock is Smaller
        self.assertIsNone(left_sub_blocks[-1].get_correct_parity())
        self.assertIsNotNone(left_sub_blocks[-1].get_parent_block())
        self.assertIsNone(left_sub_blocks[-1].get_parent_block().get_right_sub_block())
        self.assertEqual(left_sub_blocks[-1].get_end_index() - left_sub_blocks[-1].get_start_index(), 5)
        self.assertFalse(self.reconciliation._correct_parity_is_known_or_can_be_inferred(left_sub_blocks[-1]))



    def test_correct_parity_is_known_or_can_be_inferred_get_correct_parity_possible_infer(self):
        shuffle = Shuffle(self.noisy_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)

        block_size = 15

        blocks = Block.create_covering_blocks(self.noisy_key, shuffle, block_size)

        for block in blocks:
            self.reconciliation._schedule_ask_correct_parity(block, False)

        self.reconciliation._service_pending_ask_correct_parity_secure_optimized()

        left_sub_blocks = []

        right_sub_blocks = []

        for block in blocks:
            left_sub_block = block.create_left_sub_block()
            left_sub_blocks.append(left_sub_block)
            right_sub_block = block.create_right_sub_block()
            right_sub_blocks.append(right_sub_block)
            self.reconciliation._schedule_ask_correct_parity(left_sub_block, False)

        # All Left Sub Blocks now know its correct parity But do not know about right sibling
        self.reconciliation._service_pending_ask_correct_parity_secure_optimized()

        correct_parities = [1,1,1,0,1,1,1]


        for right_block, parity in zip(right_sub_blocks[:-1],correct_parities[:-1]):
            self.assertIsNone(right_block.get_correct_parity())
            self.assertIsNotNone(right_block.get_parent_block())
            self.assertIsNotNone(right_block.get_parent_block().get_left_sub_block())
            self.assertIsNotNone(right_block.get_parent_block().get_left_sub_block().get_correct_parity())
            self.assertIsNotNone(right_block.get_parent_block().get_correct_parity())
            self.assertEqual(right_block.get_end_index() - right_block.get_start_index(), 7)
            self.assertTrue(self.reconciliation._correct_parity_is_known_or_can_be_inferred(right_block))
            self.assertEqual(right_block.get_correct_parity(), parity)


        #Final Sub Block is Smaller
        self.assertIsNone(right_sub_blocks[-1].get_correct_parity())
        self.assertIsNotNone(right_sub_blocks[-1].get_parent_block())
        self.assertIsNotNone(right_sub_blocks[-1].get_parent_block().get_left_sub_block())
        self.assertIsNotNone(right_sub_blocks[-1].get_parent_block().get_left_sub_block().get_correct_parity())
        self.assertIsNotNone(right_sub_blocks[-1].get_parent_block().get_correct_parity())
        self.assertEqual(right_sub_blocks[-1].get_end_index() - right_sub_blocks[-1].get_start_index(), 5)
        self.assertTrue(self.reconciliation._correct_parity_is_known_or_can_be_inferred(right_sub_blocks[-1]))
        self.assertEqual(right_sub_blocks[-1].get_correct_parity(), correct_parities[-1])

    def test_try_correct_simple_key(self):

        key_size = 10

        correct_key = Key.create_random_key(key_size)

        error_rate = 0.1

        #One Bit Wrong
        noisy_key = correct_key.copy(error_rate, Key.ERROR_METHOD_EXACT)

        estimated_bit_error_rate = 0.5

        mock_classical_channel = MockClassicalChannel(correct_key)

        reconciliation = Reconciliation("original", mock_classical_channel, noisy_key,
                                             estimated_bit_error_rate)


        shuffle = Shuffle(noisy_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)

        block_size = 2

        blocks = Block.create_covering_blocks(noisy_key, shuffle, block_size)

        for block in blocks:
            reconciliation._schedule_ask_correct_parity(block, False)

        reconciliation._service_all_pending_work(True)

        self.assertEqual(blocks[-1].get_error_parity(), Block.ERRORS_ODD)

        reconciliation._try_correct(blocks[-1], False, True)

        self.assertEqual(str(correct_key), str(noisy_key))

    def test_shuffle(self):
        print("TEST SHUFFLE")
        key_size = 2

        correct_key = Key.create_random_key(2)

        print(correct_key)

        shuffle = Shuffle(key_size, Shuffle.SHUFFLE_RANDOM)

        print(shuffle)

        shuffle_index_1 = 1
        shuffle_index_0 = 0

        key_index_of_shuffle_index_0 = shuffle.get_key_index(shuffle_index_0)
        key_index_of_shuffle_index_1 = shuffle.get_key_index(shuffle_index_1)

        print(f"Key Index of Shuffle Index 0: {key_index_of_shuffle_index_0}")
        print(f"Key Index of Shuffle Index 1: {key_index_of_shuffle_index_1}")

        print("Shuffled Key:")
        for i in range(key_size):
            key_index_of_shuffle = shuffle.get_key_index(i)
            bit = correct_key.get_bit(key_index_of_shuffle)
            print(bit,end="")

        self.assertEqual(key_index_of_shuffle_index_0, shuffle_index_1)
        self.assertEqual(key_index_of_shuffle_index_1, shuffle_index_0)


    def test_register_block_key_indexes_single(self):
        blocks = []

        shuffle = Shuffle(self.noisy_key.get_size(), Shuffle.SHUFFLE_RANDOM)

        block_1 = Block(self.noisy_key, shuffle, 0, 15, None)
        blocks.append(block_1)
        block_2 = Block(self.noisy_key, shuffle, 15, 30, None)
        blocks.append(block_2)
        block_3 = Block(self.noisy_key, shuffle, 30, 45, None)
        blocks.append(block_3)
        block_4 = Block(self.noisy_key, shuffle, 45, 60, None)
        blocks.append(block_4)
        block_5 = Block(self.noisy_key, shuffle, 60, 75, None)
        blocks.append(block_5)
        block_6 = Block(self.noisy_key, shuffle, 75, 90, None)
        blocks.append(block_6)
        block_7 = Block(self.noisy_key, shuffle, 90, 100, None)
        blocks.append(block_7)

        for block in blocks:
            self.reconciliation._register_block_key_indexes(block)

        #print(f"Block Key Indexes: {self.reconciliation._key_index_to_blocks}")

        for i in range(15):
            #In Case of Shuffling it is required to get the Shuffling Index to the Key index
            index_aux = shuffle.get_key_index(i)
            block_aux = self.reconciliation._get_blocks_containing_key_index(index_aux)[0]
            self.assertEqual(block_1, block_aux)
        for i in range(15,30):
            index_aux = shuffle.get_key_index(i)
            block_aux = self.reconciliation._get_blocks_containing_key_index(index_aux)[0]
            self.assertEqual(block_2, block_aux)
        for i in range(30,45):
            index_aux = shuffle.get_key_index(i)
            block_aux = self.reconciliation._get_blocks_containing_key_index(index_aux)[0]
            self.assertEqual(block_3, block_aux)
        for i in range(45,60):
            index_aux = shuffle.get_key_index(i)
            block_aux = self.reconciliation._get_blocks_containing_key_index(index_aux)[0]
            self.assertEqual(block_4, block_aux)
        for i in range(60,75):
            index_aux = shuffle.get_key_index(i)
            block_aux = self.reconciliation._get_blocks_containing_key_index(index_aux)[0]
            self.assertEqual(block_5, block_aux)
        for i in range(75,90):
            index_aux = shuffle.get_key_index(i)
            block_aux = self.reconciliation._get_blocks_containing_key_index(index_aux)[0]
            self.assertEqual(block_6, block_aux)
        for i in range(90,100):
            index_aux = shuffle.get_key_index(i)
            block_aux = self.reconciliation._get_blocks_containing_key_index(index_aux)[0]
            self.assertEqual(block_7, block_aux)


    def test_register_block_key_indexes_multiple(self):
        blocks = []

        shuffle = Shuffle(self.noisy_key.get_size(), Shuffle.SHUFFLE_RANDOM)

        block_1 = Block(self.noisy_key, shuffle, 0, 15, None)
        blocks.append(block_1)
        block_1_l = block_1.create_left_sub_block()
        block_1_r = block_1.create_right_sub_block()
        blocks.append(block_1_l)
        blocks.append(block_1_r)

        block_2 = Block(self.noisy_key, shuffle, 15, 30, None)
        block_2_l = block_2.create_left_sub_block()
        block_2_r = block_2.create_right_sub_block()
        blocks.append(block_2)
        blocks.append(block_2_l)
        blocks.append(block_2_r)

        block_3 = Block(self.noisy_key, shuffle, 30, 45, None)
        block_3_l = block_3.create_left_sub_block()
        block_3_r = block_3.create_right_sub_block()
        blocks.append(block_3)
        blocks.append(block_3_l)
        blocks.append(block_3_r)

        block_4 = Block(self.noisy_key, shuffle, 45, 60, None)
        block_4_l = block_4.create_left_sub_block()
        block_4_r = block_4.create_right_sub_block()
        blocks.append(block_4)
        blocks.append(block_4_l)
        blocks.append(block_4_r)

        block_5 = Block(self.noisy_key, shuffle, 60, 75, None)
        blocks.append(block_5)
        block_5_l = block_5.create_left_sub_block()
        block_5_r = block_5.create_right_sub_block()
        blocks.append(block_5_l)
        blocks.append(block_5_r)

        block_6 = Block(self.noisy_key, shuffle, 75, 90, None)
        blocks.append(block_6)
        block_6_l = block_6.create_left_sub_block()
        block_6_r = block_6.create_right_sub_block()
        blocks.append(block_6_l)
        blocks.append(block_6_r)

        block_7 = Block(self.noisy_key, shuffle, 90, 100, None)
        blocks.append(block_7)
        block_7_l = block_7.create_left_sub_block()
        block_7_r = block_7.create_right_sub_block()
        blocks.append(block_7_l)
        blocks.append(block_7_r)

        for block in blocks:
            self.reconciliation._register_block_key_indexes(block)

        #print(f"Block Key Indexes: {self.reconciliation._key_index_to_blocks}")

        shuffle_index_0 = shuffle.get_key_index(0)
        blocks_index_0 = self.reconciliation._get_blocks_containing_key_index(shuffle_index_0)
        print(f"blocks Index 0: {blocks_index_0}")
        self.assertEqual(block_1, blocks_index_0[0])
        self.assertEqual(block_1_l, blocks_index_0[1])

        shuffle_index_10 = shuffle.get_key_index(10)
        blocks_index_10 = self.reconciliation._get_blocks_containing_key_index(shuffle_index_10)
        self.assertEqual(block_1, blocks_index_10[0])
        self.assertEqual(block_1_r, blocks_index_10[1])

        shuffle_index_15 = shuffle.get_key_index(15)
        blocks_index_15 = self.reconciliation._get_blocks_containing_key_index(shuffle_index_15)
        self.assertEqual(block_2, blocks_index_15[0])
        self.assertEqual(block_2_l, blocks_index_15[1])

        shuffle_index_25 = shuffle.get_key_index(25)
        blocks_index_25 = self.reconciliation._get_blocks_containing_key_index(shuffle_index_25)
        self.assertEqual(block_2, blocks_index_25[0])
        self.assertEqual(block_2_r, blocks_index_25[1])


    def test_flip_key_bit_corresponding_to_single_bit_block(self):
        """This Tests ensures that the Cascade is performing correctly as well as the flip bit operation"""

        #Initialization of a specific bits for creation of the key
        correct_bits = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1}

        key_size = 14

        correct_key = Key(size=key_size, bits=correct_bits)

        #Initialization of a specific bits for creation of the noisy key - Actual Error Rate 14%
        noisy_bits = {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 0, 10: 1, 11: 1, 12: 1, 13: 1}

        noisy_key = Key(size=key_size, bits=noisy_bits)

        #Manually perform the Reconciliation Method to Test Cascading Effect

        print("Test Cascade Effect")

        print(f"Correct Key:\n{correct_key}")
        print(f"Noisy Key:\n{noisy_key}")

        mock_classical_channel = MockClassicalChannel(correct_key)

        #Arbitrary Value for Reconciliation Instance Creation
        estimated_bit_error_rate = 0.09

        reconciliation = Reconciliation("original", mock_classical_channel, noisy_key,
                                             estimated_bit_error_rate)

        shuffle = Shuffle(key_size, Shuffle.SHUFFLE_KEEP_SAME)

        print(f"Shuffle: {shuffle}")

        #Consider the Entire Key as an Top Block for Ease of Implementation Testing - Start First Reconciliation Division
        top_block = Block(noisy_key, shuffle, 0, 14, None)
        sub_block_left = top_block.create_left_sub_block()
        sub_block_right = top_block.create_right_sub_block()
        #Single Bit Block is created in order to Invert this single bit block which is child of the left block and verify that cascade is performed correctly
        single_bit_block = Block(noisy_key, shuffle, 4, 5, sub_block_left)

        #Save the Block Key Indexes used in Cascade Effect of method flip_key_bit_corresponding
        reconciliation._register_block_key_indexes(top_block)
        reconciliation._register_block_key_indexes(sub_block_left)
        reconciliation._register_block_key_indexes(sub_block_right)
        reconciliation._register_block_key_indexes(single_bit_block)

        #Schedule and Ask Correct Parities of Blocks
        reconciliation._schedule_ask_correct_parity(top_block, False)
        reconciliation._schedule_ask_correct_parity(sub_block_left, False)
        reconciliation._schedule_ask_correct_parity(sub_block_right, False)
        reconciliation._schedule_ask_correct_parity(single_bit_block, False)

        reconciliation._service_pending_ask_correct_parity_secure_optimized()

        #Remove All Pending Try Correct cause ask_correct_parity inserts elements which is not intended in this usecase test
        reconciliation._pending_try_correct = []

        #Having all the base Case Setup - The Next Step is to simply Use the Flip Bit Method to change bit in index 4, and the Cascade Effect Must Correct the Remaining Bit
        #Verify that in this usecase the single bit block is indeed incorrect
        self.assertEqual(single_bit_block.get_error_parity(), Block.ERRORS_ODD)

        #Perform the method flip key bit corresponding to single bit block to verify cascade
        reconciliation._flip_key_bit_corresponding_to_single_bit_block(single_bit_block, True)

        print(f"reconciliation pending ask correct parity: {reconciliation._pending_ask_correct_parity}")
        self.assertEqual(reconciliation._pending_ask_correct_parity, [])

        #Service pending try correct is performed to verify that the previous method inserts the top block in to the try correct and this method then inserts the ask_parity
        reconciliation._service_pending_try_correct(True)

        print(f"reconciliation pending ask correct parity: {reconciliation._pending_ask_correct_parity}")

        #This is done to perform all the remaining operations, asking parities and trying to correct recursively.
        #This is performed after all the previous steps to confirm the correct functioning of the Cascade Effect.
        errors_corrected = reconciliation._service_all_pending_work(False)

        print(f"errors corrected: {errors_corrected}")

        #Make Sure the Two Keys are equal at End of this method execution - Confirming that the Cascade is performing correctly.
        self.assertEqual(str(correct_key), str(noisy_key))





if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestReconciliation("test_reconciliation"))
    #suite.addTest(TestReconciliation("test_reconciliation"))
    #suite.addTest(TestReconciliation("test_register_block_key_indexes_single"))
    #suite.addTest(TestReconciliation("test_register_block_key_indexes_multiple"))
    suite.addTest(TestReconciliation("test_flip_key_bit_corresponding_to_single_bit_block"))
    """
    suite.addTest(TestReconciliation("test_reconciliation"))
    suite.addTest(TestReconciliation("test_ask_correct_parity_first_iteration"))
    suite.addTest(TestReconciliation("test_ask_correct_parity_w_create_blocks"))
    suite.addTest(TestReconciliation("test_correct_parity_is_known_or_can_be_inferred_get_correct_parity_not_asking"))
    suite.addTest(TestReconciliation("test_correct_parity_is_known_or_can_be_inferred_get_correct_parity_asking_not_parent_block"))
    suite.addTest(TestReconciliation("test_correct_parity_is_known_or_can_be_inferred_get_correct_parity_asking_has_parent_block_no_right_sibling"))
    suite.addTest(TestReconciliation("test_correct_parity_is_known_or_can_be_inferred_get_correct_parity_possible_infer"))
    """
    #suite.addTest(TestReconciliation("test_try_correct_block"))
    #suite.addTest(TestReconciliation("test_try_correct_simple_key"))
    #suite.addTest(TestReconciliation("test__flip_key_bit_corresponding_to_single_bit_block"))
    #suite.addTest(TestReconciliation("test_shuffle"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
