# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from task_config import task_config
from worlds import MultiWozEvalWorld

MASTER_QUALIF = {
    'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',
    'Comparator': 'Exists',
    'RequiredToPreview': True
}

MASTER_QUALIF_SDBOX = {
    'QualificationTypeId': '2ARFPLSP75KLA8M8DH1HTEQVJT3SY6',
    'Comparator': 'Exists',
    'RequiredToPreview': True
}

LOCALE_QUALIF_SDBOX = {
    'QualificationTypeId': '00000000000000000071',
    "Comparator": "In",
    'LocaleValues': [{'Country': "HK"}, {'Country': "US"}, {'Country': "CN"}]
}


def main():
    """This task consists of an MTurk agent evaluating a chit-chat model. They
    are asked to chat to the model adopting a specific persona. After their
    conversation, they are asked to evaluate their partner on several metrics.
    """
    argparser = ParlaiParser(False, add_model_args=True)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '-dp', '--datapath', default='./',
        help='path to datasets, defaults to current directory')

    opt = argparser.parse_args()

    # add additional model args
    opt['override'] = {
        'no_cuda': True,
        'interactive_mode': True,
        'tensorboard_log': False
    }

    # Set the task name to be the folder name
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    # append the contents of task_config.py to the configuration
    opt.update(task_config)

    mturk_agent_id = 'Tourist'

    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id]
    )

    mturk_manager.setup_server()

    try:
        mturk_manager.start_new_run()
        mturk_manager.ready_to_accept_workers()
        mturk_manager.create_hits([LOCALE_QUALIF_SDBOX])

        mturk_manager.set_onboard_function(onboard_function=None)

        # mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        # def assign_worker_roles(workers):
        #     for index, worker in enumerate(workers):
        #         worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        def assign_worker_roles(workers):
            workers[0].id = mturk_agent_id

        def run_conversation(mturk_manager, opt, workers):
            agents = workers[:]
            # workers[0].assignment_generator = assignment_generator

            world = MultiWozEvalWorld(
                opt=opt,
                agent=workers[0]
            )

            while not world.episode_done():
                print("parley")
                world.parley()

            print("save data")
            world.save_data()

            print("world shutdown")
            world.shutdown()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )

    except BaseException:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
