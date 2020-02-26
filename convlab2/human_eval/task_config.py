# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Chat and evaluate bot!'

"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will chat to a tour information bot and then evaluate that bot.'

"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog'

"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = \
    """
    (You can keep accepting new HITs after you finish your current one, so keep working on it if you like the task!)
    <br>
    
    <span id="user-goal" style="font-size: 16px;"> 
    </span>
    
    <br><br>
    Chat with the bot naturally and stick to your own goal but <b>do not trivially copy the goal descriptions into the message.</b>
    <br>
    Once the conversation is done, you will be asked to rate the bot on metrics like <b>goal accomplishment, language understanding, and response naturalness</b>.
    <br>
    There is a <b>2 min</b> time limit for each turn.
    <br>
    <br>
    - Do not reference the task or MTurk itself during the conversation.
    <br>
    <b><span style="color:red">- No racism, sexism or otherwise offensive comments, or the submission will be rejected and we will report to Amazon.</b></span>
    <br>
    <br>
    
    <script type="text/javascript">
    
    function handle_new_message(new_message_id, message) {
      var agent_id = message.id;
      var message_text = message.text
      if (displayed_messages.indexOf(new_message_id) !== -1) {
        // This message has already been seen and put up into the chat
        log(new_message_id + ' was a repeat message', 1);
        return;
      }
      log('New message, ' + new_message_id + ' from agent ' + agent_id, 1);
      displayed_messages.push(new_message_id);
      if (agent_id !== cur_agent_id) {
        add_message_to_conversation(agent_id, message_text, false);
      } else {
        add_message_to_conversation(agent_id, message_text, true);
      }
      if ("goal_text" in message) {
        var goal_text = message.goal_text
          $("#user-goal").html('<br>You are looking for information in Cambridge.<br><br><b> Your assigned goal is:</b><br><br>' + goal_text);
          log(message.goal_text, 1);
      }
      if("exceed_min_turns" in message && message.exceed_min_turns){
        exceed_min_turns = true;
        $("button#id_done_button").css("disabled", false);
        $("button#id_done_button").css("display", "");
        $("input#id_text_input").css("width", "40%");
        $(window).resize(window_resize);
      }
      if ("evaluation" in message && message.evaluation){
        $("button#id_done_button").hide()
      }
    }
    </script>
    """
