"""Template-based mock story generator. Pure functions, no I/O, no deps."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safety import age_bucket


DEFAULT_THEME = "friends"
LENGTH_PARAGRAPHS = {"short": 3, "medium": 4, "long": 5}


def _select(theme: str, age: int, seed: int) -> dict:
    """Pick a template. Falls back to the default theme if the request is unknown."""
    bucket = age_bucket(age)
    primary, alt = THEMES.get(theme, THEMES[DEFAULT_THEME]).get(bucket, THEMES[DEFAULT_THEME]["middle"])
    return alt if seed % 2 else primary


def mock_story(name: str, age: int, themes: list[str], moral: str, length: str, seed: int = 0) -> tuple[str, str]:
    name = name or "Friend"
    primary = (themes[0] if themes else DEFAULT_THEME).lower()
    extras_phrase = ", ".join(themes[1:]) if len(themes) > 1 else "the world around them"

    template = _select(primary, age, seed)
    title = template["title"].format(name=name)
    body = template["body"].format(
        name=name,
        extras=extras_phrase,
        moral=moral or "being kind to others",
    )

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    paragraphs = paragraphs[: LENGTH_PARAGRAPHS.get(length, 4)]
    return title, "\n\n".join(paragraphs)


# THEMES: theme_name -> bucket -> (primary_template, alt_template)
# Each template is {"title": str, "body": str} with .format(name, extras, moral) placeholders.
# Two variants per (theme, bucket) drive seed rotation: even seed -> primary, odd -> alt.

THEMES: dict[str, dict[str, tuple[dict, dict]]] = {
    "dinosaurs": {
        "young": (
            {
                "title": "{name} and the Gentle Giant",
                "body": (
                    "One sunny morning, {name} packed a small lunch and walked to the edge of a big green field.\n\n"
                    "There, behind the tall ferns, a very large dinosaur was sleeping. It was a Brontosaurus, and it had the kindest eyes {name} had ever seen.\n\n"
                    "The dinosaur opened one eye and whispered hello. {name} sat beside {extras}, and together they watched the clouds turn into shapes.\n\n"
                    "Before {name} went home, the dinosaur taught a small lesson: {moral}. {name} smiled, and the dinosaur smiled back."
                ),
            },
            {
                "title": "{name} and the Singing Egg",
                "body": (
                    "Behind a bush in the garden, {name} found an egg that hummed a tiny tune.\n\n"
                    "When the egg cracked open, out stepped a small dinosaur with feathers the color of sunrise.\n\n"
                    "It followed {name} through the yard, peeking at {extras} and chirping happily at the sky.\n\n"
                    "Before the day was over, the little dinosaur taught {name} a song about {moral}. {name} hummed it all through dinner."
                ),
            },
        ),
        "middle": (
            {
                "title": "{name} and the Dinosaur Who Forgot",
                "body": (
                    "{name} had heard stories about a small valley where a young dinosaur had lost its way.\n\n"
                    "When {name} arrived, the dinosaur was hiding under a wide leaf, looking very worried. Its name was Pip, and it had forgotten the path back to its family.\n\n"
                    "Together, {name} and Pip followed the sound of the river, meeting {extras} along the way. Each new friend reminded Pip of a part of the trail.\n\n"
                    "When they reached the herd, Pip's mother bowed her long neck low. She said that being brave and asking for help was the same thing as {moral}.\n\n"
                    "{name} waved goodbye and walked home, a little taller than before."
                ),
            },
            {
                "title": "{name} and the Trail of Clues",
                "body": (
                    "On a school trip to the museum, {name} noticed a fossil that didn't match the others.\n\n"
                    "The plaque said it had been found near {extras}, in a place no one visited anymore.\n\n"
                    "With permission, {name} and the class went to look. They found not just a fossil, but a small footprint, fresh as morning.\n\n"
                    "Back at school, the teacher said the most important lesson of the day was {moral}, and {name} nodded, already planning the next trip."
                ),
            },
        ),
        "older": (
            {
                "title": "The Day {name} Met the Last Diplodocus",
                "body": (
                    "The map in {name}'s grandparents' attic was old and faded, but the path it described still wound through the hills behind the house.\n\n"
                    "{name} followed it carefully, past the old oak, past the river that sang, until the trees opened onto a clearing no one else seemed to know.\n\n"
                    "There, in a circle of soft moss, stood a young Diplodocus, humming to itself. It looked up, surprised but not afraid, and explained that it had been waiting for someone curious enough to find it.\n\n"
                    "They talked for a long while. The dinosaur spoke of {extras}, and of a quiet truth: {moral}. {name} listened, and learned that some lessons grow slowly, like the tallest trees.\n\n"
                    "On the walk home, the wind carried a gentle hum, and {name} understood that the valley would always be there, waiting."
                ),
            },
            {
                "title": "{name} and the Cartographer of Old Bones",
                "body": (
                    "In the basement of the public library, {name} found a drawer marked DO NOT OPEN in friendly red letters.\n\n"
                    "Inside were hand-drawn maps, and on the oldest, a star marking a place where the bones of small dinosaurs could still be found.\n\n"
                    "{name} followed the map across the river, past {extras}, and into a quiet canyon that smelled faintly of rain and stone.\n\n"
                    "There, beneath a flat rock, lay a single bone, carefully placed, as if waiting. {name} understood, suddenly, the rule of {moral}.\n\n"
                    "On the walk home, the wind hummed through the canyon, and {name} marked the spot with a small cairn, the way old travelers used to."
                ),
            },
        ),
    },
    "robots": {
        "young": (
            {
                "title": "{name} and the Little Robot",
                "body": (
                    "Under {name}'s bed sat a small box with one blinking light.\n\n"
                    "When {name} said hello, the box opened and out climbed a tiny robot no bigger than a kitten. It beeped softly and looked up at {name} with round blue eyes.\n\n"
                    "The robot wanted to learn about {extras}. {name} showed it how to share a cookie, and the robot's light turned a happy green.\n\n"
                    "That night, the robot whispered a small idea: {moral}. {name} hugged it, and the light blinked slowly, the way hearts do when they are content."
                ),
            },
            {
                "title": "{name} and the Robot Who Sang",
                "body": (
                    "{name} found a small robot in the attic, covered in dust and humming to itself.\n\n"
                    "It wanted to learn one thing before it went to sleep: what {extras} sounded like.\n\n"
                    "{name} played a drum, the wind played a whistle, and the robot played along on a single, bright note.\n\n"
                    "When it finally closed its eyes, it whispered its favorite idea: {moral}. {name} tucked it into a small box and promised to play again tomorrow."
                ),
            },
        ),
        "middle": (
            {
                "title": "{name} Builds a Friend",
                "body": (
                    "{name} found an old instruction manual in the garage, full of diagrams for a small helper robot.\n\n"
                    "With patience, and a little help from {extras}, {name} built a robot that could listen, fetch, and ask good questions.\n\n"
                    "One afternoon, the robot asked why some days felt heavier than others. {name} thought, then answered with something close to {moral}.\n\n"
                    "The robot nodded, its antenna glowing a soft amber. From that day on, it remembered every kindness shown to it, and tried to return each one in its own small way."
                ),
            },
            {
                "title": "{name} Repairs a Memory",
                "body": (
                    "The old family robot had stopped working the way it used to, and {name} wanted to bring it back.\n\n"
                    "Inside, among the wires, {name} found a tiny chip that held a memory of {extras}.\n\n"
                    "Carefully, {name} cleaned the chip and put it back. The robot woke slowly, blinked twice, and asked, in a small voice, if {name} was still there.\n\n"
                    "'I am,' said {name}, and the robot said, 'Good. That is what {moral} means, I think.'\n\n"
                    "From that day on, the robot preferred to work in the kitchen, where the light was warm and the company steady."
                ),
            },
        ),
        "older": (
            {
                "title": "The Robot Who Wanted to Know Why",
                "body": (
                    "{name} had built many robots before, but never one that asked so many questions.\n\n"
                    "This one wanted to know why leaves change color, why people sing when they are happy, and why kindness sometimes costs the most.\n\n"
                    "So {name} took the robot on long walks through the neighborhood, into the library, and out to the field where {extras} gather in the late afternoon.\n\n"
                    "At the end of one long walk, the robot stopped and said, 'I think I understand. It sounds like {moral}.' {name} smiled and replied, 'Yes. Now you really are awake.'\n\n"
                    "They walked home together, the robot humming a tune it had learned from a passing bird."
                ),
            },
            {
                "title": "The Last Robot {name} Will Build",
                "body": (
                    "{name} had built many robots, but never one quite like this.\n\n"
                    "This one was small and quiet, with a single question programmed into its heart: what is {moral}?\n\n"
                    "{name} took it on long walks, to the garden, to the river, to the field where {extras} grow in the late summer light.\n\n"
                    "At the end of one walk, the robot stopped and said, 'I have listened enough. I think I know.' And {name} smiled, and turned the robot off, gently, knowing it was enough.\n\n"
                    "Some machines, it turns out, are built to be finished."
                ),
            },
        ),
    },
    "space": {
        "young": (
            {
                "title": "{name} and the Sleepy Moon",
                "body": (
                    "One night, the moon looked very tired, so {name} climbed up the tallest ladder in the yard to say goodnight.\n\n"
                    "The moon yawned a long, silver yawn. It was worried because it had lost a small, twinkling star.\n\n"
                    "{name} looked around and found it hiding near {extras}, glowing shyly. {name} carried it back, and the moon smiled wide.\n\n"
                    "It whispered a sleepy thank you, and a little secret: {moral}. {name} climbed back down and slept soundly until morning."
                ),
            },
            {
                "title": "{name} and the Star That Was Lost",
                "body": (
                    "One night, the stars were all in the sky except one, and the moon was worried.\n\n"
                    "{name} climbed to the roof with a small ladder and a kind voice and called up to the lost star.\n\n"
                    "The star, it turned out, had been hiding behind {extras}, feeling shy about shining.\n\n"
                    "{name} told it that even small lights matter, and that {moral} is true for stars too. The star, blushing silver, returned to the sky, where it has shone happily ever since."
                ),
            },
        ),
        "middle": (
            {
                "title": "{name} Among the Stars",
                "body": (
                    "When the rocket's engines hummed, {name} felt the seat press gently against their back, and the sky began to fall away.\n\n"
                    "Beyond the clouds, a small planet waved hello. On it lived {extras}, who had been waiting for a visitor who asked polite questions.\n\n"
                    "{name} stayed for tea and listened to the planet sing. Its favorite song was about {moral}, and it offered to teach {name} the chorus.\n\n"
                    "When the rocket came home, {name} could still hear the song in the wind, like a small bell that never quite stops ringing."
                ),
            },
            {
                "title": "{name} and the Map of Quiet Places",
                "body": (
                    "On the spaceship, the captain kept a special map marked with the quietest places in the galaxy.\n\n"
                    "{name} had been chosen, this trip, to be the one who visited them. Each stop held a small lesson.\n\n"
                    "On the second planet, the wind sang about {extras}. On the third, the rocks hummed a song about {moral}.\n\n"
                    "When {name} returned, the captain asked what {name} had learned. 'That quiet places,' said {name}, 'are not empty. They are full of answers we are in a hurry to miss.'\n\n"
                    "The captain nodded, and pinned the map to the wall, where the whole crew could see it."
                ),
            },
        ),
        "older": (
            {
                "title": "The Comet {name} Named",
                "body": (
                    "Once every hundred years, a small comet swings close enough to be seen from a quiet rooftop in the hills.\n\n"
                    "{name} had been waiting since the first cold night of autumn, notebook in hand, watching the sky with patient eyes.\n\n"
                    "When the comet finally appeared, trailing light like a long bright ribbon, {name} whispered a name to it. The comet, as if it had been waiting too, dipped slightly toward the rooftop.\n\n"
                    "From that night on, astronomers far away would point up and speak of {name}'s comet, a gentle reminder of {moral}.\n\n"
                    "And on clear nights, if you listened, you could almost hear the wind repeat the name, soft as a secret."
                ),
            },
            {
                "title": "{name} and the Telescope at the Edge of the Field",
                "body": (
                    "It was a small telescope, older than {name}'s grandparents, and it lived in a wooden shed at the edge of a field.\n\n"
                    "On the clearest night of the year, {name} carried it out, set it down, and looked up.\n\n"
                    "The sky was not empty. It was full of small stories, each one a star, each star a reminder of {extras} and of the slow, patient rule of {moral}.\n\n"
                    "{name} stayed for a long time, until the cold became part of the night, and then went inside, leaving the telescope pointing up, just in case the stars wanted to look back."
                ),
            },
        ),
    },
    "fantasy": {
        "young": (
            {
                "title": "{name} and the Door That Wasn't There",
                "body": (
                    "Behind the bookshelf in the hallway, there was a door that only appeared on rainy afternoons.\n\n"
                    "{name} opened it and stepped into a garden where flowers hummed and rabbits wore tiny vests.\n\n"
                    "The garden asked {name} to help find a missing petal, and {name} searched with the help of {extras} until they found it resting on a sleeping snail.\n\n"
                    "The garden bowed its leaves and shared its favorite rule: {moral}. {name} waved, and the door closed softly behind, waiting for next rain."
                ),
            },
            {
                "title": "{name} and the Cat Who Could Read",
                "body": (
                    "In the corner of the library, a small cat sat reading a very tiny book.\n\n"
                    "{name} sat down beside it, and the cat, in a polite voice, read a story about {extras}.\n\n"
                    "When the story ended, the cat yawned and said its favorite rule was {moral}.\n\n"
                    "{name} agreed, and the two of them chose the next book together, the way friends do."
                ),
            },
        ),
        "middle": (
            {
                "title": "The Map {name} Drew",
                "body": (
                    "{name} had always been good at drawing maps of places that did not exist, and one Tuesday, one of them began to be true.\n\n"
                    "The map showed a forest where the trees had names, and a river that ran both up and down. At the center stood a small castle made entirely of books.\n\n"
                    "{name} walked the path they had drawn, meeting {extras} who offered riddles and warm bread. Each answer pointed to the next part of the journey.\n\n"
                    "At the castle, the librarian said the only rule worth keeping was {moral}, and gave {name} a blank page in return.\n\n"
                    "{name} took the page home and began to draw the next map, smiling."
                ),
            },
            {
                "title": "{name} and the Well That Sang",
                "body": (
                    "At the bottom of the garden, an old stone well had begun to hum, very softly, in the late afternoons.\n\n"
                    "{name} leaned in and heard, far down, a small voice singing about {extras}.\n\n"
                    "It wasn't scary, only lonesome, and so {name} dropped a small message into the well, in a waterproof bottle, that said, 'You are not alone.'\n\n"
                    "The next day, the well sang a new song, this one about {moral}. {name} smiled, and went to find a grown-up to share the news."
                ),
            },
        ),
        "older": (
            {
                "title": "The Spell {name} Didn't Mean to Cast",
                "body": (
                    "It started, as it often does, with a word spoken at the wrong moment in the right place.\n\n"
                    "{name} had been reading in the attic when the old spellbook fell open to a page about gentle things, and a small word slipped out, half-remembered.\n\n"
                    "The room filled with soft light, and from the page stepped a creature made of paper and ink, who bowed politely and asked to be useful.\n\n"
                    "Together, they tidied the attic, finding {extras} and a long-lost letter. The creature explained that even small magic follows the rule of {moral}.\n\n"
                    "When the work was done, the creature stepped back into the page, leaving behind a single feather and the feeling that attics, too, have hearts."
                ),
            },
            {
                "title": "{name} and the Apprentice Mapmaker",
                "body": (
                    "There was a mapmaker in the village who could draw a road that didn't exist yet, and {name} had been chosen to help.\n\n"
                    "The first map was for a baker who wanted a shortcut to the river. The second was for a child who dreamed of {extras}.\n\n"
                    "Each map took a whole afternoon, drawn in ink made from tea, and each one ended with the same small note in the corner: {moral}.\n\n"
                    "By the end of the week, the village had a dozen new paths, and {name} had learned that making a map is mostly a way of listening."
                ),
            },
        ),
    },
    "adventure": {
        "young": (
            {
                "title": "{name} and the Big Hill",
                "body": (
                    "There was a hill at the end of {name}'s street, and {name} had never climbed all the way to the top.\n\n"
                    "One bright morning, {name} packed a snack, waved to {extras}, and started up the path.\n\n"
                    "The hill was bigger than it looked, but {name} took small steps and rested when tired. At the top, a small flag waited, flapping hello.\n\n"
                    "It said, in friendly letters, {moral}. {name} planted it firmly and ran down to tell everyone."
                ),
            },
            {
                "title": "{name} and the Bridge of Leaves",
                "body": (
                    "A small stream had gotten too wide to jump, and {name} wanted to reach the other side.\n\n"
                    "With some rope and {extras}, {name} built a small bridge out of leaves and twigs.\n\n"
                    "It wobbled, but it held, and on the other side, {name} found a tiny garden no one had ever seen.\n\n"
                    "The garden, in a whisper, said its only rule was {moral}. {name} nodded, and crossed back carefully, already planning to return."
                ),
            },
        ),
        "middle": (
            {
                "title": "{name} and the Forgotten Path",
                "body": (
                    "Grandma had told {name} about a path behind the orchard that led to a waterfall no one visited anymore.\n\n"
                    "{name} found it on a cool afternoon, half-covered in leaves, and followed it carefully. The forest grew quiet and listening.\n\n"
                    "Along the way, {name} met {extras} who each shared a clue: a feather pointing north, a smooth stone to mark a turn, a song to hum at a fork.\n\n"
                    "The waterfall was small, but its sound filled the clearing. {name} sat and remembered the rule of every long walk: {moral}.\n\n"
                    "On the way home, the path seemed a little less forgotten, as if it were glad to be walked again."
                ),
            },
            {
                "title": "{name} and the Cave of Echoes",
                "body": (
                    "On a long hike, {name} found a cave that repeated every word, soft as a sigh.\n\n"
                    "Inside, {name} spoke carefully, telling the cave about {extras} and asking it to remember.\n\n"
                    "The cave answered, in many voices, with the same quiet rule: {moral}.\n\n"
                    "When {name} stepped out into the afternoon, the wind seemed to carry a little of the echo with it, and the hike home felt shorter than it should have."
                ),
            },
        ),
        "older": (
            {
                "title": "The Trail {name} Blazed",
                "body": (
                    "The maps in the library said no trail ran between the two valleys, but the old stories said otherwise.\n\n"
                    "{name} packed light, said goodbye to {extras}, and started into the woods with a notebook, a compass, and a great deal of stubbornness.\n\n"
                    "The first day was hard. The second was harder. On the third, {name} found the cairn left by someone long ago, and followed its markers to the pass.\n\n"
                    "Standing between the two valleys, {name} understood what the old trails always tried to teach: {moral}.\n\n"
                    "On the way down, {name} marked each turn with a small stone, so the next traveler would not have to start from nothing."
                ),
            },
            {
                "title": "{name} and the Trail of Small Signs",
                "body": (
                    "The old guidebook said no trail ran over the mountain, but {name} trusted the small signs.\n\n"
                    "A red thread tied to a branch. A stack of smooth stones. A single word painted on a rock in friendly letters: onward.\n\n"
                    "Each sign pointed toward the next, and each one hinted at {extras} and at the patient rule of {moral}.\n\n"
                    "At the summit, {name} found a small notebook in a tin box, full of names. {name} added one, closed the tin, and started down, leaving the signs for whoever came next."
                ),
            },
        ),
    },
    "animals": {
        "young": (
            {
                "title": "{name} and the Brave Little Fox",
                "body": (
                    "In the meadow near {name}'s house, there lived a small fox with a bright orange tail.\n\n"
                    "The fox was shy, but it followed {name} home one afternoon, peeking around the garden gate.\n\n"
                    "{name} shared a sandwich, and the fox shared a secret: it knew where {extras} liked to play on warm days.\n\n"
                    "Before it left, the fox whispered a kind truth: {moral}. {name} nodded, and the fox vanished into the long grass with a flick of its tail."
                ),
            },
            {
                "title": "{name} and the Cat in the Window",
                "body": (
                    "On the third floor of the apartment building, a small cat watched the street from a wide window.\n\n"
                    "{name} waved up at it every morning on the way to school, and the cat, in its careful way, waved back.\n\n"
                    "One Saturday, {name} was invited up for milk and a story. The cat told a quiet one about {extras} and ended with a small idea: {moral}.\n\n"
                    "{name} went home humming the story, and the cat returned to its window, content to be both a friend and a watcher."
                ),
            },
        ),
        "middle": (
            {
                "title": "The Animal School {name} Found",
                "body": (
                    "Behind the old barn, past the broken fence, there was a clearing where animals held a small school of their own.\n\n"
                    "{name} discovered it on a quiet morning, and the rabbit teaching arithmetic waved {name} over to join.\n\n"
                    "The lesson that day was about {extras}, and the importance of listening before speaking. The owl nodded, the mouse took notes, and the fox sharpened a small pencil.\n\n"
                    "When the bell rang, the rabbit told {name} the school's only rule, which was the same as {moral}.\n\n"
                    "{name} promised to return, and the clearing seemed to settle a little deeper into its peaceful hum."
                ),
            },
            {
                "title": "{name} and the Lost Rabbit",
                "body": (
                    "On the way home from the park, {name} noticed a small rabbit sitting very still by the fence.\n\n"
                    "It had no collar and looked unsure of where to go. {name} knelt down, very gently, and asked where it lived.\n\n"
                    "The rabbit blinked and hopped toward {extras}, and {name} followed, slowly, until they reached a small warren at the edge of the garden.\n\n"
                    "The mother rabbit thanked {name} in the way rabbits do, which is to thump once, gently. The rule of that small family, it seemed, was {moral}."
                ),
            },
        ),
        "older": (
            {
                "title": "{name} and the Council of the Wood",
                "body": (
                    "On the night of the first frost, the animals of the wood hold a council under the old cedar, and this year, they had invited a guest.\n\n"
                    "{name} arrived wrapped in a warm coat and sat very still as the deer, the badger, and the small, ancient owl spoke of the season ahead.\n\n"
                    "They talked of {extras}, of food to share, and of the long winter that was beginning to settle over the hills.\n\n"
                    "At the end, the owl turned to {name} and said, 'We have always taught our young the same thing: {moral}.'\n\n"
                    "{name} thanked them and walked home through the frost, carrying the council's small, unspoken gift: the sense of being trusted by a place."
                ),
            },
            {
                "title": "{name} and the Geese at the Reservoir",
                "body": (
                    "Each autumn, the geese stopped at the reservoir on their way south, and {name} liked to sit and watch them.\n\n"
                    "They argued and arranged themselves, called to each other in long, patient sentences, and took turns resting.\n\n"
                    "One afternoon, a single goose waddled up to {name} and stood nearby, not afraid, only tired. {name} sat still, breathing slowly, the way you do around a wild thing.\n\n"
                    "After a long while, the goose returned to the flock, and {name} understood, without being told, the rule of {moral}. Some lessons arrive on cold wind, and ask nothing in return."
                ),
            },
        ),
    },
    "ocean": {
        "young": (
            {
                "title": "{name} and the Friendly Whale",
                "body": (
                    "On the beach where {name} liked to collect shells, a great gray whale came very close to the shore one morning.\n\n"
                    "It opened one eye, the size of a dinner plate, and looked at {name} with surprising gentleness.\n\n"
                    "{name} offered a small fish, and the whale sang a low, happy note that made the seagulls pause mid-flight. It pointed with a fin toward {extras} in the tide pools.\n\n"
                    "Before swimming away, the whale hummed a single idea: {moral}. The waves kept humming it long after it had gone."
                ),
            },
            {
                "title": "{name} and the Crab Who Shared",
                "body": (
                    "In a tide pool near the rocks, a small crab was guarding a single shiny pebble.\n\n"
                    "{name} sat down beside it, and the crab, after a long pause, offered to share the pebble for just one minute.\n\n"
                    "It was heavier than it looked, and very smooth. {name} held it carefully, then gave it back.\n\n"
                    "The crab, pleased, told {name} the rule of the tide pool: {moral}. {name} bowed politely, and the tide came in, as it always does, right on time."
                ),
            },
        ),
        "middle": (
            {
                "title": "The Lighthouse {name} Tended",
                "body": (
                    "When the old lighthouse keeper hurt his ankle, {name} offered to tend the light for the week.\n\n"
                    "Each night, {name} climbed the spiral stairs and watched the beam swing across the dark water, guiding boats home through {extras} and weather.\n\n"
                    "On the third night, a small voice from the radio said a fishing boat was in trouble. {name} kept the light steady, and the boat found its way back.\n\n"
                    "The keeper said, when he returned, that the light only really works when the keeper remembers {moral}.\n\n"
                    "{name} nodded, and from then on, the spiral stairs never felt quite as long."
                ),
            },
            {
                "title": "{name} and the Sail That Wouldn't Stay Full",
                "body": (
                    "On a borrowed boat, {name} and an older cousin tried to catch a clean wind out of the harbor.\n\n"
                    "The sail flapped, the boom swung, and the wind seemed to be playing a game.\n\n"
                    "At last, the wind settled, the sail filled, and the boat slipped past the breakwater into open water. {name} felt the salt on their lips and laughed.\n\n"
                    "The cousin, in a calm voice, said the sea always teaches the same lesson: {moral}. {name} nodded, and held the tiller steady until the sun began its slow slide toward the water."
                ),
            },
        ),
        "older": (
            {
                "title": "{name} and the Map of Tides",
                "body": (
                    "In the library at the edge of the harbor, there was a chart older than anyone alive, drawn in ink made of seaweed and quiet patience.\n\n"
                    "{name} had been studying it for months, learning the names of currents and the moods of the moon.\n\n"
                    "One morning, the harbor master asked for help guiding a sailboat through {extras}, and {name} read the chart aloud, calm and clear, until the sailboat slipped safely into the bay.\n\n"
                    "The harbor master said the chart's margin, in faded handwriting, held a single rule: {moral}.\n\n"
                    "{name} copied the rule into a small notebook and walked home along the seawall, listening to the tide remember it too."
                ),
            },
            {
                "title": "{name} and the Bell Buoy at Midnight",
                "body": (
                    "The harbor had a small bell buoy that rang only when the tide was high and the night was very still.\n\n"
                    "{name} had wanted to hear it for years, and one August night, finally, did.\n\n"
                    "Standing on the dock, with the wind cool and {extras} somewhere in the dark, the bell rang twice, and the water answered, softly, from far out.\n\n"
                    "{name} stood very still and thought about patience, and about {moral}, and about the fact that some sounds wait years for the right kind of listener."
                ),
            },
        ),
    },
    "friends": {
        "young": (
            {
                "title": "{name} and the New Friend",
                "body": (
                    "On the first day of a new week, a quiet child moved into the house next door to {name}.\n\n"
                    "{name} brought over a small cup of lemonade and a drawing of {extras}, just to say hello.\n\n"
                    "The new child smiled, slowly, like a flower opening, and shared a favorite crayon in return.\n\n"
                    "Before they waved goodbye, the new child said, in a small voice, that {moral} was the best thing to remember when you are new.\n\n"
                    "{name} agreed, and the two of them began to plan tomorrow."
                ),
            },
            {
                "title": "{name} and the Treehouse Rule",
                "body": (
                    "In the back garden, {name} and a friend built a small treehouse out of blankets and imagination.\n\n"
                    "They agreed on one rule before they climbed up: that everyone who came inside was safe, and that {extras} were always welcome.\n\n"
                    "The rule, they decided, was the same as {moral}, just with a roof.\n\n"
                    "They shared cookies and stories until the streetlights came on, then climbed down carefully, agreeing to build the second room tomorrow."
                ),
            },
        ),
        "middle": (
            {
                "title": "{name} and the Club of Small Wonders",
                "body": (
                    "{name} and three friends had started a club whose only rule was to notice small, good things.\n\n"
                    "On Tuesdays, they met under the big maple and traded observations: a bird building a nest, a librarian's quiet laugh, a perfectly round pebble.\n\n"
                    "This week, the topic was {extras}, and each friend brought something to show. There were stories, drawings, and a small, folded letter.\n\n"
                    "The club's secret rule, which everyone knew but no one said out loud, was {moral}.\n\n"
                    "When the meeting ended, each member carried a little more of the afternoon home with them."
                ),
            },
            {
                "title": "{name} and the Lemonade Stand",
                "body": (
                    "On a long, hot Saturday, {name} and two friends set up a small lemonade stand at the end of the driveway.\n\n"
                    "The first customer was the mail carrier. The second was a tired dog, who got water instead. The third was a neighbor who told a long, kind story about {extras}.\n\n"
                    "By the end of the afternoon, the stand had made enough for ice cream, and the friends had made enough for a long memory.\n\n"
                    "They closed up, sharing the ice cream on the porch, and agreed that {moral} was the best part of the day."
                ),
            },
        ),
        "older": (
            {
                "title": "{name} and the Long Letter",
                "body": (
                    "When {name}'s best friend moved across the country, they decided to write letters the old way, on paper, with pens that needed dipping.\n\n"
                    "The first letters were easy, full of news and small jokes about {extras}. The later ones grew longer, slower, more honest.\n\n"
                    "In one letter, the friend wrote that distance is not the same as absence, and that {moral} is what keeps a friendship warm across any number of miles.\n\n"
                    "{name} read it twice, then sat down to write back, the kettle on, the cat asleep, the room full of the particular quiet of two people thinking of each other.\n\n"
                    "Some letters, it turns out, are small rooms where friends can sit together, no matter how far."
                ),
            },
            {
                "title": "{name} and the Notebook We Share",
                "body": (
                    "On the first day of the new school year, {name} and a friend agreed to share a small notebook, passing it back and forth between classes.\n\n"
                    "Each wrote a little — a question, a sketch, a small observation about {extras}, a quiet worry.\n\n"
                    "By the end of the term, the notebook was full, and so were the two friends, in a way that had very little to do with the lessons in class.\n\n"
                    "They agreed that the notebook's only rule, the one that mattered most, was {moral}, and that some friendships, like some books, are better when they're written in more than one handwriting."
                ),
            },
        ),
    },
}
