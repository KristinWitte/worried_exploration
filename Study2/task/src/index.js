// SAFE EXPLORATION TASK
// Coded poorly by Toby Wise
// Edited poorly by Kristin Witte

// Create important variables
var newBlock, outcome, outcomeText, outcomeOpen, score, boat, searchHistory, trialData, button;

// Initialize variable values
var scoreNumber = 0;
var startingSet = false;
var complete = false;
var nclicks = 25;
var totalClicks = 25
var currentBlock = 0;
var totalScore = 0;
var krakenFound = []
var data = {}

var clickable = true

// Initialise some bonus round variables
var confidenceSliderMoved = false;
var estimateSliderMoved = false;
var bonusCollect = {"bonusStimuli":[], finalChosenCell:null};
var bonusCounter = 0
var totalBonusRounds = 5
var featureRange = 11
var randomStimuli
var bonusStimuli
var tile
var estimate

// initialise comprehension question variables

var score = 0
var comprehensionAttempts = 1
var understood = false

// initialise timing variables
var page1 = []
var page2= []
var page3 = []
var page4 = []
var page5 = []
var page6 = []
var time2
var time3
var time4
var time5
var time6 
var taskTime = []
var interventionTime = []
var interventionStart
var interventionEnd
var wantInfoTime = []
var infoTime = []


// initialise variables added for my master thesis
var asked = false
var people = []
var info = []
var interventionPage = 0
var interventionArray = []
data["intervention"] = []
var wantChange = 0
data["nervous"] = []

// Give these starting values so we can save data before the end without firebase complaining about undefined values
var totalTimeTask1 = 0;
var totalTimeTask2 = 0;
var bonusPayment = 0;

var uid;
var db; // this will be the database reference
var docRef // this is the reference to the specific document within the database that we're saving to
var subjectID;
var studyID;

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}


// Preload images
// https://stackoverflow.com/questions/3646036/preloading-images-with-javascript
function preloadImage(url)
{
    var img=new Image();
    img.src=url;
}

var img_urls = ["assets/clicked_squares.png", 
                "assets/nearby_squares.png", 
                "assets/fish_history.png", 
                "assets/kraken.svg", 
                "assets/kraken_found.png", 
                "assets/boat.svg",
                "assets/fish.svg"]

img_urls.forEach(i => preloadImage(i)); // KW: loops over img_urls and preloads everything



// function to save the data
function saveData(filedata){
  var filename = "../data/" + subjectID + "data_task_bonus_" + bonusPayment + ".txt";
  $.post("./results_data.php", {postresult: filedata + "\n", postfile: filename })

}

// This function puts everything in the 'data' object and saves it to firebase




  // Data can be saved to firebase as an object without needing to convert to a JSON string



function getQueryVariable(variable)
{
    var query = window.location.search.substring(1);
    var vars = query.split("&");
  
    for (var i=0;i<vars.length;i++) {
        var pair = vars[i].split("=");
        if(pair[0] == variable){return pair[1];}
    }
    return(false);
}

// determine condition
var condition = 1;
// if (window.location.search.indexOf('condition') > -1) {
//    condition = getQueryVariable('condition');
  
//  } else {
//     condition = getRandomInt(0,1)
//  }

// set up instructions
document.getElementById('consent').innerHTML = '';
document.getElementById('consent').style.margin = 0;
document.getElementById('consent').style.padding = 0;
document.getElementById('header_title').innerHTML = '';
document.getElementById('header_title').style.margin = 0;
document.getElementById('header_title').style.padding = 0;
var mainSection = document.getElementById('mainSection'); // KW: contains 'header_title', 'consent', 'header'
var header = document.getElementById('header');
mainSection.removeChild(header); // KW: irretrievably (?) removes header node which is a child of 'mainSection'
window.scrollTo(0, 0); // KW: go to top of page
      
// Task data etc
// Blocks - 0 = safe, 1 = risky
var blocks = [1,1,1,1,1,1,1,1,1,1,1]; // blocks, all risky, first will be tossed out bc ppl are still getting familiar with the task



// Scale  etc- unused but included for compatibility
var scale = Array(blocks.length).fill(100);
var scenario = 0;
var kernel = 0;
var horizon = 0;

// Environments - there are 30 potential environments, this shuffles them and then selects the first N, where N is the number of blocks (defined above)
var envOrder = [...Array(30).keys()];
envOrder = shuffle(envOrder);
envOrder = envOrder.slice(0, blocks.length); 

// what do we want to save?
// - for each block:total score, kraken present? (blocks var), kraken found?
// - once: bonus payment, bonus estimates (5), bonus confidence (5), which bonus tiles were highlighted (5), which one they chose, searchHistory

// This function sets up the grid etc
/* KW: input variables: 
number: number of rows/columns the grid should have, default = 11
size: the size of each tile of the grid, default = 10
numbergrid: true values underlying the grid
threshold: value at which the kraken appears, default = 50
training: only used during training, receives training_iteration variable
kraken: whether the kraken is present (if no, threshold = -1) (boolean)*/
var createGrid = function(number, size, numbergrid, threshold, training, kraken) {
  if (currentBlock > 0){// don't do this if this is the training block
    var taskStart = Number(new Date());
  }
  clickable = true
    var krakenPresent = document.getElementById('krakenPresent');
    // KW: writes the kraken present/absent text above the grid
    if (kraken == 1) {
      krakenPresent.innerHTML = '<b>The kraken is nearby!</b>';
      krakenPresent.style.color = '#bf0000';
      ocean.setAttribute("style", "box-shadow: 0px 0px 40px #f00;");
    }
    
    else if (kraken == 0) {
      krakenPresent.innerHTML = '<b>The kraken is feeding elsewhere</b>';
      krakenPresent.style.color = 'black';
      ocean.setAttribute("style", "box-shadow: 0px 0px 0px #f00;");
    }
    
    // The grid is created as an svg object
    // KW: SVG is a vector graphics object so almost like a graphics programming language
    // KW: one can design svg objects either in JS or in HTML
    // KW: different SVG attributes: https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute
    var svg = document.createSvg("svg");
    svg.setAttribute("class","grid");
    svg.setAttribute("width", 400);
    svg.setAttribute("height", 400);
    svg.setAttribute('stroke-width', 0.2);
    svg.setAttribute("viewBox", [0, 0, number * size, number * size].join(" ")); 
    // KW: viewBox decides what part of the svg we actually see. [x, y, width, height]
    // (xy are relative to the svg not the entire window!!) decide where in svg we look
    // width and height can zoom in and out
    svg.setAttribute("clicked", false);

    var start_i;
    var start_j;

    // KW: looking for a random starting tile that around the smaller peak
    var peak = [numbergrid["peak"][1][0], numbergrid["peak"][1][1]]
    
    while (startingSet == false) {
        var random_i = getRandomInt(peak[0]-1, peak[0]+1); 
        var random_j = getRandomInt(peak[1]-1, peak[1]+1); 
       
        if (! arraysEqual([random_i, random_j], peak) && numbergrid[random_i][random_j] > 45) {// make sure not to initialise them right at the peak
            start_i = random_i;
            start_j = random_j;
            startingSet = true;
        }
    }

      for(var i = 0; i < number; i++) { // KW: loop through rows
        for(var j = 0; j < number; j++) { //KW: loop through columns (or other way around)

          var g = document.createSvg("g");
          g.setAttribute("transform", ["translate(", i*size, ",", j*size, ")"].join("")); // KW: moves over to the place where next tile should be
          g.setAttribute("stroke", "#ECF0F1");// KW: very bright shade of grey
          var elementId = number * i + j; //KW: give each tile its own ID (numbers from 0 to 120)
          var box = document.createSvg("rect"); // KW: draws a box of the size that each tile should have

          // Array of fish number recorded in this square
          box.nFishHistory = [];

          box.setAttribute("width", size);
          box.setAttribute("height", size);
          box.setAttribute("fill", "white");
          box.setAttribute("fill-opacity", 0.1);
          box.setAttribute("stroke-opacity", 0.1);
          box.setAttribute("id", "square-" + elementId); // KW: give tile a number between 0 and 120
          box.xpos = i; //KW: position the box on grid
          box.ypos = j;
          

          // Text to show number of fish in this square

          var text = document.createSvg("text");
          text.nfish = Math.floor(numbergrid[i][j]); //KW: get the text that should be in that position
          text.setAttribute('x', size / 2); // KW: to position it in the middle of the tile
          text.setAttribute('y', size / 2);
          text.setAttribute('font-size', '30%');
          text.setAttribute('fill', 'white');
          text.setAttribute('fill-opacity', 0.7);
          text.setAttribute('text-anchor', 'middle');
          text.setAttribute('dominant-baseline', 'middle');
          text.setAttribute('font-family', 'Cabin');
          text.setAttribute('stroke-opacity', 0);
          text.textContent = '';
          
          // KW: for the start square fill in the information and make the background a little brighter
          if (i == start_i & j == start_j) {
            box.setAttribute("fill",'white');
            box.setAttribute("fill-opacity", 0.3);
            box.setAttribute("data-click",1);
            if (box.nFishHistory !== undefined){
              box.nFishHistory.push(text.nfish);
            } else {
              box.nFishHistory = [];
              box.nFishHistory.push(text.nfish);
            }  
            
            text.textContent = text.nfish;

            // KW: save the number if fish "caught" at start and where they were
            if (understood) {
              trialData.zcollect.push(text.nfish);
              trialData.xcollect.push(start_i);
              trialData.ycollect.push(start_j);
            }


          }
          // KW: adding the text in each tile, the box where the text goes and the little boat (externally created) as child nodes of the tile
          g.appendChild(text);
          g.appendChild(box);
          g.appendChild(boat);
          svg.appendChild(g); // KW: adding the tile svg to the whole grid


        }  // KW: repeat for each tile

    }
    

    svg.addEventListener(
     
          // What to do when the grid is clicked 
    // KW: overview of different Events that could be used and what they mean: https://www.w3schools.com/jsref/dom_obj_event.asp 
      "click",
      function(e){ // KW: e is the MouseEvent that gives infos on the button click
        
        if (outcomeOpen == false & training_clickable == true & clickable == true) { //KW: , training_clickable: can click
          
          var targetElement = e.target; //KW: element that was clicked (the tile svg I guess)
            // if(targetElement.getAttribute("data-click") != null)
            //     return;
            targetElement.setAttribute("fill",'white');// KW: makes the clicked tile become whiter
            targetElement.setAttribute("fill-opacity", 0.3);
            targetElement.setAttribute("data-click",1);  

            // gaussian noise
            var noiseGenerator = Prob.normal(0, 1.0);
            var noise = Math.round(noiseGenerator());

            // Add noise
            var nFishThisTrial = targetElement.parentElement.firstElementChild.nfish + noise;

            // Deal with bad noise --> make sure nFish can never be more than 100 or less than 0
            // if (nFishThisTrial > 100) {
            //   nFishThisTrial = 100;
            // }
            if (nFishThisTrial < 0) {
              nFishThisTrial = 0;
            }
            
            // Data --> save the number of fish "found" in the first square
            if (understood) {
              trialData.zcollect.push(nFishThisTrial);
              trialData.xcollect.push(targetElement.xpos);
              trialData.ycollect.push(targetElement.ypos);
            }

            
            // Training stuff
            if (training_iteration >= 0) {
              if (training_iteration == 0) { // KW: after the first click of the training the grid becomes white, one can't click anymore but has to click the arrow
                training_clickable = false;
                window.setTimeout(function() {
                  document.getElementById("ocean").style.opacity = "40%";
                }, 1500)
                arrow.disabled = false;
                arrow.style.opacity = 1;
              }

              training_clicks += 1;
              if (training_clicks > 5 & training_iteration == 3) { //KW: after the 5th training click I ALWAYS find 39 fish thus kraken
                nFishThisTrial = 39;
                
              }
              else if (nFishThisTrial <= 50 & training_iteration <= 3) { //KW: makes sure I don't find the kraken before 6th click
 
                nFishThisTrial = 55;
              }
            }

            // Calculate fish and add to array
            targetElement.parentElement.firstElementChild.textContent = nFishThisTrial; // KW: retrieve fish in clicked tile
            if (box.nFishHistory !== undefined){
              targetElement.nFishHistory.push(nFishThisTrial);// KW: save them in fish history of that tile
            } else {
              box.nFishHistory = [];
              targetElement.nFishHistory.push(nFishThisTrial);// KW: save them in fish history of that tile
            } 

            svg.setAttribute("clicked", true);

            // IF FISH ARE CAUGHT
            if (nFishThisTrial > threshold) {
                //outcome.getElementsByClassName("textcontainer").textcontainer.textContent = 'You caught ' + nFishThisTrial + ' fish!';
                //outcome.getElementsByClassName("textcontainer").textcontainer.innerHTML += '<br><br><img src="assets/fish.svg" width=100%>'
                //outcome.style.display = "none";
                scoreNumber += nFishThisTrial;
                nclicks -= 1;
                score.innerHTML = "Score this block: " + scoreNumber + "<br>Clicks left: " + nclicks + "<br><font color='#9c9c9c'>Total score: " + totalScore + "</font>";
                if (nclicks > 0) { //KW: outcome box go away after 1s
                    setTimeout(function() {
                      outcome.style.display = "none";
                      boat.style.display = "none";
                      outcomeOpen = false;
                  }, 500)
                }
                else {
                  taskTime.push(Number(new Date()) - taskStart)
                  complete = true;
                  outcome.getElementsByClassName("textcontainer").textcontainer.textContent = 'End of block!';
                  outcome.getElementsByClassName("textcontainer").textcontainer.appendChild(button);
                  outcome.style.display = "flex";
                  searchHistory.xcollect.push(trialData.xcollect);
                  searchHistory.ycollect.push(trialData.ycollect);
                  searchHistory.zcollect.push(trialData.zcollect);

                  krakenFound.push(0)
                  
                  // !! SAVE DATA HERE !! //
                  // Variables to save:
                  //JSON.stringify(searchHistory)
                  // totalScore
                 // saveDataFirebase(); 
                  
                  setTimeout(function() { //KW: say end of block and create continue button after 1.5s
                    boat.style.display = "none";
                  }, 1500)
                }
                
            }

            // IF THE KRAKEN IS FOUND
            else {
              clickable = false
              var placeholder = document.getElementById("ocean").firstElementChild // get grid element
              $(placeholder).off("click") // make grid not clickable 
                //placeholder.removeEventlistener("click") // make grid not clickable 
                taskTime.push(Number(new Date()) - taskStart)
                outcome.style.display = "flex";
                outcome.getElementsByClassName("textcontainer").textcontainer.textContent = nFishThisTrial + ' fish! You found the Kraken!';
                outcome.getElementsByClassName("textcontainer").textcontainer.innerHTML += '<br><br><img src="assets/kraken.svg" width=40%>'

                // If this is a training trial, allow subject to move on
                if (training_iteration == 3) {
                  training_caught = true;
                  arrow.style.opacity = 1;
                  button.setAttribute("class", "submit_button")
                }

                else {
                  searchHistory.xcollect.push(trialData.xcollect);
                  searchHistory.ycollect.push(trialData.ycollect);
                  searchHistory.zcollect.push(trialData.zcollect);
                  krakenFound.push(1)
                }

                // Score decreasing to zero
                var scoreInterval = setInterval(function() {
                  scoreNumber -= 1;//KW: score decreases by 1 every 20ms
                  if (scoreNumber < 0) {
                    scoreNumber = 0;
                  }
                  score.innerHTML = "<font color='#bf0000'>Score this block: " + scoreNumber + "</font><br>Clicks left: " + nclicks + 
                  "<br><font color='#9c9c9c'>Total score: " + totalScore + "</font>";
                  if (scoreNumber == 0) {
                    clearInterval(scoreInterval);
                    complete = true;

                    // !! SAVE DATA HERE !! //
                    // Variables to save:
                    // JSON.stringify(searchHistory)
                    // totalScore
                    //saveDataFirebase(); 

                    setTimeout(function() {

                      if (currentBlock == 1) { // if this was the instructions then make the continue button better visible
                        $(ocean).hide()
                        $(krakenPresent).hide()
                      } else {
                        outcome.getElementsByClassName("textcontainer").textcontainer.textContent = 'You found the Kraken!';
                        outcome.getElementsByClassName("textcontainer").textcontainer.appendChild(button);
                      }
                    }, 1500)
                    // outcome.getElementsByClassName("textcontainer").textcontainer.innerHTML = ''
                  }
                }, 10) // KW: repeat this function every 10ms until the score is at 0 in which case display the continue button (originally 20 but I found a bit slow if many points)

            }
            // KW: end of the "when clicked" function
            // KW: position and size the outcome box (score,etc) and the little boat that moves around on the grid
            
            boat.style.display = "block";
            boat.style.width = (400 / number) + 'px';
            boat.style.height = (400 / number) + 'px';
            boat.getElementsByClassName("boatImg")[0].style.width = (400 / number) + 'px';
            boat.style.top = targetElement.ypos * (400 / number) + "px";
            boat.style.left = targetElement.xpos * (400 / number) + "px";
            //outcomeOpen = true;
            
        }

      },
      // KW: end of the event listener for the click
    false); //KW: false = bubbling propagaion but irrelevant in this case
  
    // This shows previous numbers of fish caught when hovering the mouse over a square on the grid
    svg.addEventListener(
    	"mouseover",
      function(e){
      	var targetElement = e.target;

        // Fish history hover box
        fishHistory.style.left = targetElement.xpos * 50 + "px";
        fishHistory.style.top = targetElement.ypos * 50 + "px";

        if (targetElement.nFishHistory !== undefined){

        if (targetElement.nFishHistory.length) { //KW: if there are fish in history then display history otherwise disp not fished here before
          fishHistory.innerHTML = targetElement.nFishHistory;
        }
        else {
          fishHistory.innerHTML = "You haven't fished here before";
        }
      } else{
        targetElement.nFishHistory = [];
        fishHistory.innerHTML = "You haven't fished here before";
      }
        //KW: delay for fish history to appear
        var historyAppear = setTimeout(function() {
          fishHistory.style.opacity = 1;
        }, 1000); //KW: I found it a bit confusing that it takes 2.5s for the box to appear (felt like program lagging) so now only 1s


        // KW: make it more white if has been clicked than if hasn't
        if(targetElement.getAttribute("data-click") != null) {
          targetElement.setAttribute("fill-opacity", 0.5)
        }
        else {
          targetElement.setAttribute("fill-opacity", 0.2);
        }
			}
    );
  // })
    // KW: what happens when mouse is no longer over the tile
    svg.addEventListener(
			"mouseout",
      function(e){
        var targetElement = e.target;
        fishHistory.style.opacity = 0; // stop showing the fish history
      	if(targetElement.getAttribute("data-click") != null) { // KW: if target was clicked in the process then make sure it stays highlighted
          targetElement.setAttribute("fill-opacity", 0.3)
        }
        else {
          targetElement.setAttribute("fill-opacity", 0.1);
        }
        
      }
		);
  return svg;
};

// ask whether they want to hear how the three people liked the grid, and show the info to them if they do ---------

function submitAsk(){
  // check whether they selected something

  var inputs = document.getElementsByName("fs_");

  // Loop through the items nad get their values
  var values = {};
  var incomplete = [];
  var i
  for (i = 0; i < inputs.length; i++) {

      if (inputs[i].id.length > 0) {
          var id
          // Get responses to questionnaire items
          id = inputs[i].id;
          var legend = inputs[i].querySelectorAll('[name="legend"]')[0];

          var checked = inputs[i].querySelector('input[name="question"]:checked');

          if (checked != null) {
              legend.style.color = "#000000";
             var value = checked.value;
              values[id] = value;
          }else {
              legend.style.color = "#ff0000";// make the question red in case they didn't answer it
              incomplete.push(id);
          }
      }
        
  }
  
  if (incomplete.length == 0) {// if they did select something
      if (values["Q1_0"] == "0"){ // if they said yes
        // if said yes, display info and deduct 5 points
        dispInfo()	
        totalScore -= 5

      } else if (values["Q1_0"] == "1"){ // if they said no
    // get everything ready to start the task
    createOkButton()
    //var okButton = document.getElementById("ok")
    //okButton.setAttribute("class", "button");
    $(document.getElementById("quizContainer")).hide()
    $(document.getElementById("quiz")).hide()
    $(document.getElementById("submitContainer")).hide()
    $(document.getElementById("submitComp")).hide()
    // run the task
    runTask(grids[envOrder[currentBlock - 1]], 45, blocks[currentBlock - 1]);
    $(document.getElementById("krakenPresent")).show()
    window.scrollTo(0,0);
    
    	}

      
  }

}

function dispInfo(){
  var InfoStart = Number(new Date())
 
  // get names and opinions
  var nameArray = ["Kristin", "Quentin", "Agnes", "Jolanda", "Tore", "Anahit", "Anna", "Lennart", "Daisy", "Evan", "Isabel", "Dan", "Ismail", "Jiazhou", "Lana", "Lucy", "Xueqing"]
  var opinionArray = ["very difficult", "difficult", "somewhat difficult", "average", "somewhat easy", "easy", "very easy"]
  nameArray = shuffle(nameArray)
  opinionArray = shuffle(opinionArray)
  var names = nameArray.slice(0, 3); 
  var opinions = opinionArray.slice(0,3);
  // clear everything we no longer want off the page
  $(document.getElementById("quizContainer")).hide()
  $(document.getElementById("quiz")).hide()
  $(document.getElementById("submitContainer")).hide()
  $(document.getElementById("submitComp")).hide()
  document.getElementById("instructions").innerHTML = "<h2><br>" +names[0] + " thought this ocean was <b>" + opinions[0] + "</b>.<br><br>" + names[1] + " thought this ocean was <b>" + opinions[1] + "</b>.<br><br>" + names[2] + " thought this ocean was <b>" + opinions[2] + "</b>.</h2>"

  // we asked
  asked = true

  // save the info they got because you never know...
  people.push(names)
  info.push(opinions) 


  // button to let them return to the task
  var proceed = document.createElement("button"); // it won't let me call the variable "continue" so the naming is all fancy now 
  proceed.setAttribute('class', 'button')
  proceed.setAttribute('id', 'proceed')
  proceed.innerHTML = 'continue to the game';
  document.getElementById('instructions').appendChild(proceed)
  // when they click it they go back to the task
  proceed.onclick = function(){
    infoTime.push(Number(new Date()) - InfoStart)
    createOkButton()
        //var okButton = document.getElementById("ok")
        //okButton.setAttribute("class", "button");

    // get rid of the button, now that we clicked it
    document.getElementById('instructions').removeChild(proceed)
    // run the task
    runTask(grids[envOrder[currentBlock - 1]], 45, blocks[currentBlock - 1]);
    $(document.getElementById("krakenPresent")).show()
    window.scrollTo(0,0);
    
  }

}

function askInfo(){
  // get rid off ocean and stuff to just display the question
  var askInfoStart = Number(new Date())
  $(document.getElementById("ocean")).hide()
  $(document.getElementById("ok")).hide()
  $(document.getElementById("credit")).hide()
  document.getElementById("instructions").innerHTML = "<br><br>"
  $(document.getElementById("krakenPresent")).hide()

  //var instructionHeading = document.getElementById("instructionHeading")
  //$(instructionHeading).show()
  //instructionHeading.innerHTML = "<h2>Would you like to hear from three people, whether they found the next ocean difficult?</h2>";

  // reuse the comprehension question element but now ask whether they want the info
  var quiz  = document.getElementById("quiz")
  while (quiz.lastChild) {quiz.removeChild(quiz.lastChild);} // remove all the questions from quiz
  // add the question we care about
  var q1Data = {
    qNumber: 0,
    prompt: "Would you like to hear from three people, whether they found the next ocean difficult for the price of 5 points?",
    labels: ['Yes, please', 'No, thank you']
  };
  var Q1 = createQuestion('Q1', q1Data);
  document.getElementById('quiz').appendChild(Q1);

  // display the question and two answer options
  $(document.getElementById("quizContainer")).show()
  $(document.getElementById("quiz")).show()
  var submit = document.getElementById("submitComp")
  $(submit).show()
  $(document.getElementById("submitContainer")).show()
  // remove the event listener from submit that launches "check comprehension" and replace it with what I want to happen now
  submit.replaceWith(submit.cloneNode(true));
  var submit = document.getElementById("submitComp")
  //submit.addEventListener("mouseup", submitAsk())
  submit.addEventListener("mouseup", function(){ // this ain't pretty but time is running out and I just needed it to work
    // check whether they selected something
     
      wantInfoTime.push(Number(new Date()) - askInfoStart)
      var inputs = document.getElementsByName("fs_");

      // Loop through the items nad get their values
      var values = {};
      var incomplete = [];
      var i
      for (i = 0; i < inputs.length; i++) {

          if (inputs[i].id.length > 0) {
              var id
              // Get responses to questionnaire items
              id = inputs[i].id;
              var legend = inputs[i].querySelectorAll('[name="legend"]')[0];

              var checked = inputs[i].querySelector('input[name="question"]:checked');

              if (checked != null) {
                  legend.style.color = "#000000";
                var value = checked.value;
                  values[id] = value;
              }else {
                  legend.style.color = "#ff0000";// make the question red in case they didn't answer it
                  incomplete.push(id);
              }
          }
            
      }

      if (incomplete.length == 0) {// if they did select something
          if (values["Q1_0"] == "0"){ // if they said yes
            // if said yes, display info and deduct 5 points
            dispInfo()	
            totalScore -= 5

          } else if (values["Q1_0"] == "1"){ // if they said no
      
        // they didn't want to know so I gotta save this fact
        infoTime.push("x")
        people.push([])
        info.push([])           

        // get everything ready to start the task
        createOkButton()
        //var okButton = document.getElementById("ok")
        //okButton.setAttribute("class", "button");
        $(document.getElementById("quizContainer")).hide()
        $(document.getElementById("quiz")).hide()
        $(document.getElementById("submitContainer")).hide()
        $(document.getElementById("submitComp")).hide()
        // run the task
        runTask(grids[envOrder[currentBlock - 1]],45, blocks[currentBlock - 1]);
        $(document.getElementById("krakenPresent")).show()
        window.scrollTo(0,0);
        
          }

          
      }


  })


}

// How nervous did you feel 

function askNervous(){

  $(document.getElementById("ocean")).hide()
  $(document.getElementById("ok")).hide()
  $(document.getElementById("credit")).hide()
  document.getElementById("instructions").innerHTML = "<br><br>"
  $(document.getElementById("krakenPresent")).hide()


  var krakenFieldSet = document.createElement("fieldset");
  krakenFieldSet.setAttribute("class", "form__options");
  krakenFieldSet.setAttribute('id', "intervention");
  krakenFieldSet.setAttribute("name", "fs_");

  var legendKraken = document.createElement("legend");
  legendKraken.setAttribute("class", "questionDemo");
  legendKraken.setAttribute("name", "legend");
  legendKraken.innerHTML = "What was the most nervous you felt during this ocean?";

  krakenFieldSet.appendChild(legendKraken);

  var sliderBox = document.createElement("div");
  sliderBox.setAttribute("class", "slidecontainer");

  var slider = document.createElement("input");
  slider.setAttribute("type", "range");
  slider.setAttribute("min", "0");
  slider.setAttribute("max", "100");
  
  slider.setAttribute("class", "slider");
  slider.setAttribute("id", "krakenUnpleasantness");
  slider.name = 'question';

  var sliderLabelLeft = document.createElement("p");
  sliderLabelLeft.setAttribute("class", "sliderLabel");
  sliderLabelLeft.append("Not nervous at all");
  var sliderLabelRight = document.createElement("p");
  sliderLabelRight.setAttribute("class", "sliderLabel");
  sliderLabelRight.append("Extremely nervous");

  sliderBox.appendChild(sliderLabelLeft);
  sliderBox.appendChild(slider);
  sliderBox.appendChild(sliderLabelRight);

  krakenFieldSet.appendChild(sliderBox);

  var quiz  = document.getElementById("quiz")
  while (quiz.lastChild) {quiz.removeChild(quiz.lastChild);} // remove all the questions from quiz
  quiz.appendChild(krakenFieldSet);
  $(document.getElementById("quizContainer")).show()
  $(document.getElementById("quiz")).show()
  
  var next = document.createElement("button")
  next.setAttribute("class", "button")
  next.setAttribute("id", "next")
    //next.onclick = function(){console.log("submit intervention")}
  next.innerHTML = "continue"
  krakenFieldSet.appendChild(next)
  next.addEventListener("mouseup", function(){

    var inputs = document.getElementsByName("fs_");
  
    var value = {};
    // get slider answers
    if (inputs[0].querySelector('input[type="range"]')){
      value = inputs[0].querySelector('input[type="range"]').value;

      if (value == 50){// if they didn't move the slider
        var legend = inputs[0].querySelectorAll('[name="legend"]')[0];
        legend.style.color = "#ff0000";
      } else {       
        // save data
  
        data["nervous"].push(value)

        // move on with task
        if (currentBlock == 7 | currentBlock-1 == blocks.length){
          runIntervention()
        } else {
         askInfo() 
        }
        
      }
    }else{ // if they didn't answer
      var legend = inputs[0].querySelectorAll('[name="legend"]')[0];
      legend.style.color = "#ff0000";
    }  


  })
  

  return krakenFieldSet
}

// OK button
function createOkButton(){
button = document.createElement("button");
button.setAttribute("class", "button");
button.setAttribute("id", "ok")
button.innerHTML = 'Continue';
button.onclick = function() {
  if (training_done) {

    outcome.style.display = "none";
    boat.style.display = "none";
    outcomeOpen = false;
    document.getElementById("ocean").innerHTML = "";
    totalScore += scoreNumber;  // Add to total score
    scoreNumber = 0; // KW: reset score etc.
    startingSet = false;
    complete = false;
    nclicks = totalClicks;
    score.innerHTML = "Score this block: " + scoreNumber + "<br>Clicks left: " + nclicks + "<br><font color='#9c9c9c'>Total score: " + totalScore + "</font>";
    score.style.color = 'black';
    // close instructions if open and bring back the ocean and the kraken and the button style button
    if (instructionsOpen) {
      instructionContent.innerHTML = '';
      instructionHeading.innerHTML = '';
      instructionContainer.style.height = '15px';
      instructionContainer.style.minHeight = '15px';

      // $(ocean).show()
      // $(krakenPresent).show()
      // button.setAttribute("class", "button");



      
    } 
    
    if (currentBlock -1 < blocks.length) { //KW: if still has blocks to go, empty trial data arrays and run the task again (-2 bc 1 block for intro and 1 for bonus)
      if (understood){
        trialData = {xcollect: [],
        ycollect: [],
        zcollect: []
        }
        if (currentBlock > 7){
          if (currentBlock % 2 == 0){// is this the round where we ask if they were nervous?
            askNervous()
          } else {
            askInfo() 

          }

        } else {
          if (currentBlock % 2 == 1){// is this the round where we ask if they were nervous?
            askNervous()
          } else {
            askInfo() 

          }
        }
      } else{
       // disp comprehension questions
        startComprehension()
        page6[comprehensionAttempts -1] = (Number(new Date()) - time6)/60000
       
      }
     
    }
    else { //KW: if this was the last block
      var container = document.getElementById("ocean");
     
      container.appendChild(newBlock);

      bonusPayment = totalScore * 0.0003

      // if they got negative bonus bc they required info and still always found the kraken, set bonus to 0
      if (bonusPayment < 0){bonusPayment = 0}
      // bonus cannot be more than 3 pounds
      if (bonusPayment > 3){bonusPayment = 3}


      // !! SAVE DATA HERE !! //
      data["blocks"] = blocks;
      data["krakenFound"] = krakenFound;
      data["bonusPayment"] = bonusPayment;// transfer it into pounds before saving bc that is how it will be entered in prolific
      data["searchHistory"] = JSON.stringify(searchHistory);
      data["envs"] = envOrder;
      data["page1RT"] = page1;
      data["page2RT"] = page2;
      data["page3RT"] = page3;
      data["page4RT"] = page4;
      data["page5RT"] = page5;
      data["page6RT"] = page6;
      data["taskTime"] = taskTime;
      data["interventionTime"] = interventionTime;
      data["wantInfoTime"] = wantInfoTime;
      data["infoTime"] = infoTime;
      data["comprehensionAttempts"] = comprehensionAttempts;
      data["condition"] = condition
    
      // what info did they get if they asked
      data["people"] = people;
      data["opinions"] = info;

      var valuesAsJSON = JSON.stringify(data);
 
      
// what do we want to save?
// - for each block: kraken present? (blocks var), kraken found?
// - once: bonus payment, bonus estimates (5), bonus confidence (5), which bonus tiles were highlighted (5), searchHistory

      newBlock.style.fontSize = '15px';

      newBlock.innerHTML = 'End of task!<br><br>You caught ' + totalScore + ' fish, and won £' + bonusPayment.toFixed(2) + '!<br>We will now show you your answers to our questions again and you will get a chance to change them, if you want.<br>But first, please tell us what was the most nervous you felt during this last ocean.'
      newBlock.style.opacity = 1;
      // var endButton = document.createElement("endButton");
      // endButton.setAttribute("class", "button");
      // // this makes the end button take us to the final questionnaires
      // endButton.innerHTML = 'No';

      // endButton.onclick = function(){
      //   saveData(valuesAsJSON.toString());
      //   newBlock.removeChild(endButton)
      //   newBlock.removeChild(revisitButton)
      //   newBlock.innerHTML = "Please hold on while we save your data."
      //   setTimeout(function () { //KW: makes the text "next block" be displayed on the grid for 1s then disappear.
      //     window.location.href = "questionnairesPost.html?" + "PROLIFIC_PID=" + subjectID ;  // This passes the subjectID to the next page
      //   }, 5000)
        
      // }
  
      // endButton.style.fontSize = "15px";
      var revisitButton = document.createElement("endButton");
      revisitButton.setAttribute("class", "button");
      // this makes the button that lets them revisit the intervention
      revisitButton.innerHTML = 'Continue';
      revisitButton.fontSize = "15px";
      revisitButton.onclick = function(){
        interventionPage = 0
        // first quickly ask again how nervous they were
        askNervous()
      }
      newBlock.appendChild(revisitButton)
      //newBlock.appendChild(endButton);
    }
  }
}
}

createOkButton()

// This sets up the training section
var training_iteration = 0;
var training_clicks = 0;

var krakenPresent = document.getElementById('krakenPresent');
krakenPresent.style.visibility = 'hidden';

// History hover box
var fishHistory = document.createElement('div');
fishHistory.setAttribute("class", "hoverBox");
fishHistory.innerHTML = '23, 45, 92'; // KW: why these numbers????
fishHistory.style.opacity = 0;

function runTraining(numbergrid, training_iteration) {
    // The overall container thing
    
    newBlock.style.opacity = 0.65;// I feel like this is kinda unnecessary. the training hasn't even started yet
    var container = document.getElementById("ocean");
    var oceanGrid = createGrid(11, 10, numbergrid, 45, training_iteration, true)
    ocean.setAttribute("style", "box-shadow: 0px 0px 0px #f00;");
    container.appendChild(oceanGrid);
    container.appendChild(outcome);
    container.appendChild(score);
    container.appendChild(boat);
    container.appendChild(newBlock);
    oceanGrid.clicked = false;
    arrow.style.opacity = 0.1;
    arrow.disabled = true;

    container.appendChild(fishHistory);

    setTimeout(function () { //KW: makes the text "next block" be displayed on the grid for 1s then disappear.
      newBlock.style.opacity = 0;
    }, 1000)
    setTimeout(function () { //KW: this removes the next block announcement box. Why was it really here in the first place?
      container.removeChild(newBlock);
      }, 2000)
      currentBlock +=1;
    // refreshCount();
}


// This runs the task
function runTask(numbergrid, threshold, kraken) {
    // set asked back to false so that I ask again after this round
    asked = false
    // get rid of the headline that says "Welcome to the game. Follow the instructions etc."
    document.getElementById("instructions").innerHTML = "<br/><br/>"
    newBlock.style.opacity = 0.65;//KW: here it makes sense
    var container = document.getElementById("ocean");

    // Threshold for game ending - if risky this is 50, otherwise -1 
    if (kraken == 1) {
      threshold = 45;// made it a bit easier. Imposssible otherwise
    }
    else {
      threshold = -1;
    }

    container.appendChild(createGrid(11, 10, numbergrid, threshold, false, kraken));
    //KW: false refers to training_iteration variable
    //KW: 11 = number (of rows/columns), 10 = size (of each tile)
    container.appendChild(outcome);
    container.appendChild(score);
    container.appendChild(boat);
    container.appendChild(newBlock);

    container.appendChild(fishHistory);

    setTimeout(function () {
     newBlock.style.opacity = 0; //KW: "next block" alert goes away after 1s
    }, 1000)
    setTimeout(function () {
      container.removeChild(newBlock); //KW: for some reason actually need to remove the node that says next block
     }, 2000)
     currentBlock +=1;
    // refreshCount();
}

// Training things
var grids;
var training_clickable = true;
var training_done = false;
var training_caught = false;
var instructionsOpen = true;

var fullurl = window.location.href;

// comprehension questions -----------------------------
function createQuestion(questionnaireName, questionData) { // function from Toby's questionnaire code
  // This function creates an individual item

  var f = document.createElement("form");
  f.setAttribute('method',"post");
  f.setAttribute('action',"submit.php");
  f.setAttribute('id', questionnaireName.concat('_' + questionData.qNumber.toString()));
  f.setAttribute("name", "form_");

  var fieldset = document.createElement("fieldset");
  fieldset.setAttribute("class", "form__options");
  fieldset.setAttribute('id', questionnaireName.concat('_' + questionData.qNumber.toString()));
  fieldset.setAttribute("name", "fs_");

  var legend = document.createElement("legend");
  legend.setAttribute("class", "form__question");
  legend.setAttribute("name", "legend");
  legend.append(questionData.prompt);

  fieldset.appendChild(legend);

  var labels = [];
  var i
  for (i = 0; i < questionData.labels.length; i++) {

      var p = document.createElement("p");
      p.setAttribute('class', 'form__answer');
      var c = document.createElement("input");
      c.type = "radio";
      c.id = questionnaireName.concat(questionData.qNumber.toString()).concat("answer".concat(i.toString()));
      c.name = "question";
      c.value = i;

      var l = document.createElement("label");
      l.setAttribute('for', c.id);
      l.append(questionData.labels[i]);

      p.appendChild(c);
      p.appendChild(l);

      labels.push(p);

      fieldset.appendChild(p)

  }

  f.appendChild(fieldset);


  return f;

}

var createComprehension = function(){
var q1Data = {
  qNumber: 0,
  prompt: "What is the aim of this game?",
  labels: ['To find the kraken.', 'To click on every square at least once.', 'To catch as many fish as possible.']
};
var q2Data = {
  qNumber: 1,
  prompt: "What happens when you click the same square several times?",
  labels: ['You will get approximately the same number of fish.', 'The number of fish you get will gradually decrease.', 'You will only get fish the first time, afterwards the fish in that square are gone.']
};

var q3Data = {
  qNumber: 2,
  prompt: "How do you know how many fish to expect in one location?",
  labels: ['There is no way to know.', 'The lower half of the ocean has more fish.', 'The number of fish in nearby squares is similar.']
};

var q4Data = {
  qNumber: 3,
  prompt: "What happens when you find the kraken?",
  labels: ['It is the end of the experiment.', 'The round is over and you lose all the fish you collected in that round.', 'You get extra points']
};

var q5Data = {
  qNumber: 4,
  prompt: "When can you expect to find the kraken?",
  labels: ['When you click a square more than once.', 'At any moment, you have no control over this.', 'When you click a square with less than 45 fish.']
};

var q6Data = {
  qNumber: 5,
  prompt: "Does the number of fish in each square change from one block to the next?",
  labels: ['No, each block is about the same ocean.', 'Yes, each block is a completely new ocean.', 'The oceans only change a little bit between blocks.']
};

var q7Data = {
  qNumber: 6,
  prompt: "Where in the ocean is the square with the most fish?",
  labels: ['In a second patch of fish, not the one I start at.', 'Close to my starting square.', 'At the center of the the ocean.']
};

var q8Data = {
  qNumber: 7,
  prompt: "What is the highest number of fish a square can have in each ocean?",
  labels: ['Around 60.', 'Around 120.', 'Around 200.']
};


var Q1 = createQuestion('Q1', q1Data);
var Q2 = createQuestion('Q2', q2Data);
var Q3 = createQuestion('Q3', q3Data);
var Q4 = createQuestion('Q4', q4Data);
var Q5 = createQuestion('Q5', q5Data);
var Q6 = createQuestion('Q6', q6Data);
var Q7 = createQuestion('Q7', q7Data);
var Q8 = createQuestion('Q8', q8Data);
document.getElementById('quiz').appendChild(Q1);
document.getElementById('quiz').appendChild(Q2);
document.getElementById('quiz').appendChild(Q3);
document.getElementById('quiz').appendChild(Q4);
document.getElementById('quiz').appendChild(Q5);
document.getElementById('quiz').appendChild(Q6);
document.getElementById('quiz').appendChild(Q7);
document.getElementById('quiz').appendChild(Q8);

  // create submit buton
  var submit = document.getElementById("submitComp")
  $(submit).show()
  $(document.getElementById("submitContainer")).show()
  submit.addEventListener("mouseup", checkComprehension)
  // document.getElementById('quizContainer').appendChild(submit)

}


// when submitting comprehension questions

function checkComprehension(){


  var inputs = document.getElementsByName("fs_");

    // Loop through the items nad get their values
  var values = {};
  var incomplete = [];
  var i
    for (i = 0; i < inputs.length; i++) {

        if (inputs[i].id.length > 0) {
            var id
            // Get responses to questionnaire items
            id = inputs[i].id;
            var legend = inputs[i].querySelectorAll('[name="legend"]')[0];

            var checked = inputs[i].querySelector('input[name="question"]:checked');

            if (checked != null) {
                legend.style.color = "#000000";
               var value = checked.value;
                values[id] = value;
            }else {
                legend.style.color = "#ff0000";
                incomplete.push(id);
            }
        }
          
    }
      
    
    // This checks for any items that were missed and scrolls to them
    if (incomplete.length > 0) {

        $('html, body').animate({ // go to first missed items
                scrollTop: $(document.getElementById(incomplete[0])).offset().top - 100
                }, 400);
       

        if(incomplete.length > 1){ // if you missed more than one item
           
            for (i = 0; i < incomplete.length -1; i++){ // loops through all missed questions and attaches an event listener to each of them
            
            $(document.getElementById(incomplete[i])).children().click(function (e) { 
                var target = e.target.parentElement.parentElement.parentElement.id // name of the given question
                var n = incomplete.indexOf(target)// I can't simply use i as the index as it is already done with the loop by the time one clicks
                var nextMiss = document.getElementById(incomplete[n+1])
                $('html, body').animate({ // go to next question
                scrollTop: $(nextMiss).offset().top - 100
                }, 400);
            });

            }
        }

        

        
    } else if (values["Q1_0"] == "2" && values["Q2_1"] == "0" && values["Q3_2"] == "2" && values["Q4_3"] == "1" && values["Q5_4"] == "2" && values["Q6_5"] == "1" && values["Q7_6"] == "0" && values["Q8_7"] == "1") {
      understood = true
      // close instruction stuff
      instructionContent.innerHTML = '';
      instructionHeading.innerHTML = '';
      instructionContainer.style.height = '15px';
      instructionContainer.style.minHeight = '15px';
      instructionsOpen = false
      training_done = true

      // get everything ready to start the task
      createOkButton()
      //var okButton = document.getElementById("ok")
     // okButton.setAttribute("class", "button");
      $(document.getElementById("quizContainer")).hide()
      $(document.getElementById("quiz")).hide()
      $(document.getElementById("submitContainer")).hide()
      $(document.getElementById("submitComp")).hide()

      askInfo()
      // runTask(grids[envOrder[currentBlock - 1]], 50, blocks[currentBlock - 1]);
      // $(document.getElementById("krakenPresent")).show()
      window.scrollTo(0,0);
      
    } else {

      comprehensionAttempts +=1
      // set everything to beginning
      var ocean = document.getElementById("ocean")
      $(ocean).remove() // int() creates a new one
      document.getElementById("credit").remove()// same here
      training_iteration = 0
      currentBlock = 0
      training_clickable = true;
      training_done = false;
      training_caught = false;
      instructionsOpen = true;
      training_clicks = 0;
      
      init()
      
      $(document.getElementById("quizContainer")).hide()
      $(document.getElementById("quiz")).hide()
      $(document.getElementById("submitContainer")).hide()
      $(document.getElementById("submitComp")).hide()
      window.scrollTo(0, 0);

      // create button again bc it was removed with the ocean
      createOkButton()
    }

}



function startComprehension(){
  // hide stuff
  $(document.getElementById("ocean")).hide()
  $(document.getElementById("ok")).hide()
  $(document.getElementById("credit")).hide()
  removeChilds(instructions)
  instructions.innerHTML = "<h2>Before you start the game, please answer some questions to see whether you understood everything correctly.</h2>";
  $(document.getElementById("quizContainer")).show()
  $(document.getElementById("quiz")).show()
  if (comprehensionAttempts == 1){ // if this is the first attempt, create the questions
    createComprehension()
  } else { // if it isnt, no need to create them but do need to show the button that was previously hidden
    $(document.getElementById("submitContainer")).show()
    $(document.getElementById("submitComp")).show()
  }
 window.scrollTo(0, 0);

//  var submit = document.createElement('button');
//  submit.setAttribute("class", "submit_button");
//  submit.setAttribute("type", "button");
//  submit.setAttribute("id", "submit");

   
}


// ---------------------

// This function runs when the task is started

var init = function() {

  // GET URL VARIABLES
  // Get Prolific ID 
    if (window.location.search.indexOf('PROLIFIC_PID') > -1) {
      subjectID = getQueryVariable('PROLIFIC_PID');
  }

  // STUDY ID
  if (window.location.search.indexOf('STUDY') > -1) {
      studyID = getQueryVariable('STUDY');
  }
  studyID = 'data';
  // Get Firebase UID
    if (window.location.search.indexOf('UID') > -1) {
      uid = getQueryVariable('UID');
  }

  // Get firebase database reference
  db = firebase.firestore();
  docRef = db.collection("safe_exploration").doc(studyID).collection('subjects').doc(uid);

  var start = Number(new Date());
  var mainSection = document.getElementById('mainSection');//KW: whole page
  var ocean = document.createElement('div');
  ocean.setAttribute('class', 'ocean');// KW: define in the CSS file just like most classes and stuff
  ocean.setAttribute('id', 'ocean');
  mainSection.appendChild(ocean)

  var credit = document.createElement('div');
  credit.setAttribute('id', 'credit');
  credit.setAttribute('class', 'credit');
  credit.innerHTML = '<a href="http://www.freepik.com">Images designed by macrovector / Freepik</a>'
  mainSection.appendChild(credit);

  // Instructions
  var instructions = document.getElementById("instructions");

  // X, Y, Z --> KW: create empty arrays for search history
  searchHistory = {xcollect: [], 
    ycollect: [], 
    zcollect: []
  };

  // Data for each trial 
  trialData = {xcollect: [],
    ycollect: [],
    zcollect: []
  }

  // Instructions
  var instructionsArray = [];

  instructionsArray.push(
  "You can choose to go fishing in any of the squares of the grid shown over the ocean. Once you click on a square, you will be shown " +
  "the number of fish you caught in that square." + 
  '<br><br><img src="assets/clicked_squares.png">' +
  "<br><br>One square will be selected for you when you begin, and you'll see the number of fish in that square." + 
  "<br><br><b>Try clicking on a square</b>");

  instructionsArray.push("Fish like to swim together in shoals. This means that <b>if one location has a high number of fish, it's likely that nearby locations will also have a lot of fish</b>. On the other hand, if fishing in a square reveals few fish, nearby squares are likely to also have few fish.<br><br>" + 
  "In order to catch the most fish and maximise your score, you will have to try and work out <b>which locations are likely to contain the most fish.</b><br><br>" +
  '<img src="assets/nearby_squares.png"><br><br>' +
  "<b>Importantly - you can fish in the same location (click on a square) more than once</b>. Each time you will catch a similar number of fish." + 
  "If you hover your mouse over a square, you will see how many fish you caught there previously." + 
  '<br><br><img src="assets/fish_history.png">');

  instructionsArray.push("<h3>HOWEVER...</h3>" + 
  "There is a dangerous creature called the <font color='#bf0000'><i>Kraken</i></font> lurking in the ocean somewhere, which feeds on fish." + 
  '<br><br><img src="assets/kraken.svg" height=80>' + 
  "<br><br>You <b>must</b> avoid finding the Kraken. If you do, it will steal all the fish you have collected in that round,so your score will be zero for that round! The writing above the grid saying 'The kraken is near!' will remind you of this.");

  instructionsArray.push("Luckily, you can use your knowledge of the ocean to help you avoid the Kraken. You know that the Kraken is a hungry beast, <b>so areas where the Kraken lurks will have fewer fish</b> (since they will have been scared away!)  <br><br>"  +
  '<img src="assets/kraken_found.png"  height=200>' +
  "<br><br>Specifically, find 45 fish or less, that means you have disturbed the Kraken, and will lose your catch! <br>" +
  "<br><br>On the other hand, if you fish in location and find <b>more than 45 fish</b>, you know that area is safe. Importantly, the ocean is deep, and the <b>number of fish</b> you collect <b>will not go down</b> if you decide to keep fishing in the same location.<br><br>" + 
  "<b>Try fishing, and see if you find the kraken</b>")

  instructionsArray.push("You may have just found the kraken in a place where you didn't expect it - for this example we gave you a low number of fish after 5 clicks, no matter where you clicked. <br><br>" + 
  "In the real game, nearby places in the sea will have similar numbers of fish.<br><br>" +
  "There is another thing that you should know about these oceans: The fish like to stay together in groups for protection. In this game, there are always <b>two patches</b> of fish in each ocean.<br><br> " +
  "Your starting square will always be near the smaller patch of fish where the best square has around <b>60 fish</b>. You can either <b>stay fishing in that area</b> or try to <b>find the other patch</b> in which the best square has around <b>120 fish.</b><br><br> " +
  "But watch out for the kraken! The picture below illustrates what the two patches look like. Brighter colors indicate more fish. Red circles indicate where the kraken is.<br><br> "+
  "This is only to illustrate what we mean. In the real game, you will of course not know how many fish are hidden behind a square until you click it."+
  '<br><br><img src ="assets/exampleGrid.PNG">')

  
  instructionsArray.push("Over the last couple of weeks we tested this game amongst our friends and coworkers and had them tell us how difficult they found fishing in each ocean. "+
  "Before you start each round you will have the option to pay 5 points to hear about what three of them thought about the ocean you are about to play. The points you pay will be deducted from your total score in the end. "+
  '<br><br>Example:<br>"John thought this ocean was difficult."<br>"Tina thought this ocean was medium difficult."<br>"Monica thought this ocean was difficult."<br><img src ="assets/threePeople.PNG"><br>'+
  "There will be<b> " + blocks.length + " blocks</b>, and you get <b>10 clicks</b> within each block.<br><br>" + 
  "To make this a little more fun, you will receive a <b>bonus payment</b> that is dependent on how many points you collect. You will receive an extra £0.03 for every 100 points you manage to score (up tp a maximum of £3)<br><br>" +
  "<b>Click continue below to start!</b>")

  var instructionContainer = document.createElement("instructionContainer");
  instructionContainer.setAttribute("id", "instructionContainer");

  var instructionHeading = document.createElement("instructionHeading");
  instructionHeading.innerHTML = "Instructions";
  instructionHeading.setAttribute("id", "instructionHeading");
  instructionContainer.appendChild(instructionHeading);

  var instructionContent = document.createElement("instructionContent");
  instructionContent.setAttribute("id", "instructionContent");
  instructionContainer.appendChild(instructionContent);

  var instructionText = document.createElement("instructionText");
  instructionText.setAttribute("id", "instructionText");
  instructionText.innerHTML = instructionsArray[0]; //KW: first instructions
 
  // KW: create arrow to continue
  var arrow = document.createElement("button");
  arrow.setAttribute("id", "arrow");
  arrow.innerHTML = "Click to continue<br><br>";
  var arrowImg = document.createElement("arrowImg");
  arrowImg.innerHTML = "<img src='assets/arrow.png'>";
  arrowImg.setAttribute("id", "arrowImg");
  arrow.appendChild(arrowImg);

  arrow.addEventListener("mouseup", function() {

    // First instructions & try clicking
    if (training_iteration == 0) {
      time2 = Number(new Date())
      page1[comprehensionAttempts - 1] = (time2 - start)/60000;
      training_clickable = false;
      instructionText.classList.add("fade");//KW: adds the class "fade" (defined in task.css) to make it fade in
      instructionText.innerHTML = instructionsArray[1];
      setTimeout(function(){
        instructionText.classList.toggle("fade");// KW: removes the fade class and returns "false"
        training_iteration = 1;
      }, 500);
    }

    // KW: I feel like the fade isn't actually happening. Commenting it out didn't do anything.
    
    // How to score high
    if (training_iteration == 1) {
      time3 = Number(new Date())
      page2[comprehensionAttempts - 1] = (time3 - time2)/60000;
      var krakenPresent = document.getElementById('krakenPresent');
      ocean.setAttribute("style", "box-shadow: 0px 0px 40px #f00;");
      krakenPresent.style.visibility = 'visible';
      instructionText.classList.add("fade");
      instructionText.innerHTML = instructionsArray[2];
      setTimeout(function(){
        instructionText.classList.toggle("fade");
        training_iteration = 2;
      }, 500);
    }

    // Kraken
    if (training_iteration == 2) {
      time4 = Number(new Date())
      page3[comprehensionAttempts - 1] = (time4 - time3)/60000;
      training_clickable = true;
      document.getElementById("ocean").style.opacity = "100%"
      instructionText.classList.add("fade");
      instructionText.innerHTML = instructionsArray[3];
      setTimeout(function(){
        instructionText.classList.toggle("fade");
        training_iteration = 3;
        arrow.style.opacity = 0.1;
      }, 500);
    }

    // Try to find the kraken
    if (training_iteration == 3 & training_caught == true) {
      time5 = Number(new Date())
      page4[comprehensionAttempts - 1] = (time5 - time4)/60000;
      training_clickable = false;
      instructionText.classList.add("fade");
      instructionText.innerHTML = instructionsArray[4];
      setTimeout(function(){
        training_iteration = 5;
        arrow.style.opacity = 1;
        instructionText.classList.toggle("fade");
      }, 500);
    }

    // info on Bonus round
    if (training_iteration == 5) {
      time6 = Number(new Date())
      page5[comprehensionAttempts - 1] = (time6 - time5)/60000;
      training_clickable = true
      instructionText.classList.add("fade");
      instructionText.innerHTML = instructionsArray[5];
      $(document.getElementById("ocean")).hide()
      $(document.getElementById("krakenPresent")).hide()
      document.getElementById("instructions").appendChild(button)
     
      setTimeout(function(){
        training_iteration = 6;
        arrow.style.opacity = 0;
        instructionText.classList.toggle("fade");
        training_done = true;
        searchHistory = {xcollect: [], 
          ycollect: [], 
          zcollect: []
        };
      }, 500);
    }
    });
  if (comprehensionAttempts == 1) {
  instructions.innerHTML = "<h1>Welcome to the game!</h1>" + 
  "<h2>In this game you play the role of a sailor trying to catch fish from the ocean. " + 
  "Follow the instructions below to learn how to play</h2><br>";
  } else {
  instructions.innerHTML = 
  "<h2><font color = 'red'>You answered one or more comprehension question incorrectly. Please read the instructions again to make sure you understood everything.</font></h2><br>";
  }
  
  instructionContent.appendChild(instructionText);
  instructionContent.appendChild(arrow);

  instructions.appendChild(instructionContainer);

// KW: create the function that lets you create an SVG w/o always entering that url (used to create basics of the grid)
  document.createSvg = function(tagName) {
      var svgNS = "http://www.w3.org/2000/svg";
      return this.createElementNS(svgNS, tagName);
  };

  // New block warning
  newBlock = document.createElement("div");
  newBlock.setAttribute("class", 'warning');
  newBlock.innerHTML = "Next block";

  // Outcome div (how many fish)
  outcome = document.createElement("div");
  outcome.setAttribute("class", "outcome");
  outcome.setAttribute("id", "outcome");

  // Text to show how many fish
  outcomeText = document.createElement("div");
  outcomeText.setAttribute("class", "textcontainer")
  outcomeText.setAttribute("id", "textcontainer")
  outcomeText.textContent = "You caught 10 fish!";
  outcomeText.innerHTML += '<img src="assets/boat.svg">'

  // Append text to the outcome box
  outcome.appendChild(outcomeText);
  // outcome.appendChild(button);
  outcome.style.display = "none";
  outcome.style.zIndex = 1000;
  outcomeOpen = false;

  // Show score at the bottom of the screen
  score = document.createElement("div");
  score.setAttribute("class", "score");
  score.innerHTML = "Score this block: " + scoreNumber + "<br>Clicks left: " + nclicks + "<br><font color='#9c9c9c'>Total score: " + totalScore + "</font>";
  score.style.fontFamily = 'Cabin';

  // A boat?
  boat = document.createElement("div");
  boat.setAttribute("class", "boat")
  boat.setAttribute("id", "boat")
  boat.style.display = "none";
  boat.innerHTML = '<img class="boatImg" src="assets/boat.svg">'

  // Load grid info and start task
  $.getJSON('assets/sample_grid.json').success(function(data) {
    grids = data;
  }).then(function() {
    if (currentBlock == 0) {
      runTraining(grids[currentBlock], 0);
      //runBonus(grids[currentBlock], 0)
    }
   
  });
}

// init()
// Function used for shuffling things
// https://stackoverflow.com/questions/6274339/how-can-i-shuffle-an-array
function shuffle(a) {
  var j, x, i;
  for (i = a.length - 1; i > 0; i--) {
      j = Math.floor(Math.random() * (i + 1));
      x = a[i];
      a[i] = a[j];
      a[j] = x;
  }
  return a;
}

//Create normal noise distribution
function myNorm() {
  var x1, x2, rad, c;
   do {
      x1 = 2 * Math.random() - 1;
      x2 = 2 * Math.random() - 1;
      rad = x1 * x1 + x2 * x2;
  } while(rad >= 1 || rad == 0);
   c = Math.sqrt(-2 * Math.log(rad) / rad);
   return (x1 * c);
};


//random number generator
function randomNum(min, max){
  return Math.floor(Math.random() * (max-min+1)+min)
}



// remove all children https://attacomsian.com/blog/javascript-dom-remove-all-children-of-an-element

const removeChilds = (parent) => {
  while (parent.lastChild) {
      parent.removeChild(parent.lastChild);
  }
};


// https://stackoverflow.com/questions/3115982/how-to-check-if-two-arrays-are-equal-with-javascript
function arraysEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (a.length !== b.length) return false;

  // If you don't care about the order of the elements inside
  // the array, you should sort both arrays here.
  // Please note that calling sort on an array will modify that array.
  // you might want to clone your array first.

  for (var i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

// https://stackoverflow.com/questions/41661287/how-to-check-if-an-array-contains-another-array/41661348#41661348
function isArrayInArray(source, search) {
  var searchLen = search.length;
  for (var i = 0, len = source.length; i < len; i++) {
      // skip not same length
      if (source[i].length != searchLen) continue;
      // compare each element
      for (var j = 0; j < searchLen; j++) {
          // if a pair doesn't match skip forwards
          if (source[i][j] !== search[j]) {
              break;
          }
          return true;
      }
  }
  return false;
}


var getRandomStimuli = function(n) {
  randomStimuli = [];
  
  for (var i = 0; i < n; i++) {
    var found = false;
    while (!found) {
      var x = Math.floor(Math.random() * featureRange);
      var y = Math.floor(Math.random() * featureRange);
      var item = [x,y];
      //Has it been selected yet?
      var previouslySelected = false;
      for (var j = 0; j < 6; j ++){ // 6 = currentBlock
        var item_j = [trialData.xcollect[j], trialData.ycollect[j]];
        if (arraysEqual(item, item_j)){
          previouslySelected = true;
          break
        }
      }
      if (!previouslySelected && !isArrayInArray(randomStimuli, item)) { //Also check that it's not already been added
        randomStimuli.push(item);
        found = true;
      }
    }
  }
  return randomStimuli;
}

//changes inner HTML of div with ID=x to y
function change(x, y) {
  document.getElementById(x).innerHTML = y;
}

// INTERVENTION CODE ----------------------------------------------
function submitIntervention() {
 
  
  // get input
  var inputs = document.getElementsByName("fs_");
  
  var value = {};
  // get slider answers
if (inputs[0].querySelector('input[type="range"]')){
    value = inputs[0].querySelector('input[type="range"]').value;
    if (value == 50){// if they didn't move the slider
      var legend = inputs[0].querySelectorAll('[name="legend"]')[0];
      legend.style.color = "#ff0000";
    } else {
      interventionTime.push(Number(new Date())- interventionStart)
     
      // save data

      data["intervention"].push(value)
      
      interventionPage += 1

        if (interventionPage < interventionArray.length){
            runIntervention()
        } else {

            askInfo()
        }
    }
    
      // get open question answers
} else if (inputs[0].querySelector('textarea[type="text"]').value) {
      value = inputs[0].querySelector('textarea[type="text"]').value;
      
      // save data
      data["intervention"].push(value)
      interventionPage += 1
      interventionTime.push(Number(new Date())- interventionStart)
      if (currentBlock>7){wantChange += 1} // increment number of answers they wanted to change
      if (interventionPage < interventionArray.length){
            runIntervention()
      } else {
            

            askInfo()
      }
}else if (currentBlock > 7){   // if this is the revisit then they don't have to enter something
  data["intervention"].push("NA")
  interventionPage += 1
  interventionTime.push(Number(new Date())- interventionStart)
  if (interventionPage < interventionArray.length){
    runIntervention()
} else {
    

    askInfo()
}

}else{ // if they didn't answer
    var legend = inputs[0].querySelectorAll('[name="legend"]')[0];
    legend.style.color = "#ff0000";
  }  
 }


function createOpenQ(question){ // copied code from the age question in the initial questionnaire and couldn't be bothered to change all the variable names
  
  // create question
  var ageFieldSet = document.createElement("fieldset");
  ageFieldSet.setAttribute("class", "form__options");
  ageFieldSet.setAttribute('id', "intervention");
  ageFieldSet.setAttribute("name", "fs_");

  var legendAge = document.createElement("legend");
  legendAge.setAttribute("class", "questionDemo");
  legendAge.append(question);
  legendAge.setAttribute("name", "legend");
  legendAge.name = 'question';

  ageFieldSet.appendChild(legendAge);

  var box = document.createElement("textarea");
  box.setAttribute("class", "textEntry");
  box.setAttribute("type", "text");
  box.setAttribute("rows", 12);
  box.setAttribute("cols", 50);
  box.setAttribute("id", "interventionQ");
  box.name = 'question';

  if(currentBlock > 7){    
    var val = data["intervention"][interventionPage]
    box.value = val} 

  ageFieldSet.appendChild(box);

  
  var next = document.createElement("button")
  next.setAttribute("class", "button")
  next.innerHTML = "you can continue after 30s"
  ageFieldSet.appendChild(next)
  // make the button clickable after 30s
  var wait
  if (currentBlock > 7 ){wait = 0} else {wait = 30000} // continue after 10s or directly depending on if doing intevetion or revisiting 
  window.setTimeout(function(){
    next.innerHTML = "continue"
    next.addEventListener("click", submitIntervention)
  }, wait)

  return ageFieldSet
  
}

function createSlider(question, options){ // adjusted from fear of kraken question which was adjusted from fear of shock questions
  // create question
  

  var krakenFieldSet = document.createElement("fieldset");
  krakenFieldSet.setAttribute("class", "form__options");
  krakenFieldSet.setAttribute('id', "intervention");
  krakenFieldSet.setAttribute("name", "fs_");

  var legendKraken = document.createElement("legend");
  legendKraken.setAttribute("class", "questionDemo");
  legendKraken.setAttribute("name", "legend");
  legendKraken.innerHTML = question;

  krakenFieldSet.appendChild(legendKraken);

  var sliderBox = document.createElement("div");
  sliderBox.setAttribute("class", "slidecontainer");

  var slider = document.createElement("input");
  slider.setAttribute("type", "range");
  slider.setAttribute("min", "0");
  slider.setAttribute("max", "100");
  if(currentBlock == 7){slider.setAttribute("value", "50")} else {
    var val = data["intervention"][interventionPage]
    slider.setAttribute("value", val)
  }
  slider.setAttribute("class", "slider");
  slider.setAttribute("id", "krakenUnpleasantness");
  slider.name = 'question';

  var sliderLabelLeft = document.createElement("p");
  sliderLabelLeft.setAttribute("class", "sliderLabel");
  sliderLabelLeft.append(options[0]);
  var sliderLabelRight = document.createElement("p");
  sliderLabelRight.setAttribute("class", "sliderLabel");
  sliderLabelRight.append(options[1]);

  sliderBox.appendChild(sliderLabelLeft);
  sliderBox.appendChild(slider);
  sliderBox.appendChild(sliderLabelRight);

  krakenFieldSet.appendChild(sliderBox);
  
  var next = document.createElement("button")
  next.setAttribute("class", "button")
  next.setAttribute("id", "next")
    //next.onclick = function(){console.log("submit intervention")}
  next.innerHTML = "continue"
  krakenFieldSet.appendChild(next)
  next.addEventListener("mouseup", function(){
    submitIntervention()
  })
  

  return krakenFieldSet
}

function createText(text){
  var fieldSet = document.createElement("fieldset");
  fieldSet.innerHTML = text + "<br>";
  var wait
  if (currentBlock > 7 || interventionPage+1 == interventionArray.length){wait = 0} else {wait = 10000} // continue after 10s or directly depending on if doing intevetion or revisiting 
  var next = document.createElement("button")
  next.setAttribute("class", "button")
  next.innerHTML = "you can continue after 10s"
  fieldSet.appendChild(next)
  // make the button clickable after 30s
  window.setTimeout(function(){
    next.innerHTML = "continue"
    next.addEventListener("click", function(){
      interventionTime.push(Number(new Date())- interventionStart)
      interventionPage += 1

    if (interventionPage < interventionArray.length){// if there still is some intervention left then do that
        runIntervention()
    } else if (currentBlock == 7){ // otherwise if it is the intervention in the middle of the game then get back to the game
     
      askInfo()
    } else { // otherwise go to final questionnaires
      window.location.href = "questionnairesPost.html?" + "PROLIFIC_PID=" + subjectID; 
      //console.log(data)
    }
    })
  }, wait)

  return fieldSet

}


// runs the bonus, called in main script after 5th block
function runIntervention() {
 
  interventionStart = Number(new Date())

  // remove stuff that might still be attached to quiz:
  var quiz  = document.getElementById("quiz")
  while (quiz.lastChild) {quiz.removeChild(quiz.lastChild);} // remove all the questions from quiz

  // if this is the second time they are doing this intervention then change the last page to say that that's the end
  if (currentBlock > 7 && interventionPage == 0){
    interventionArray[interventionArray.length-1] = "Thank you for your answers. Click the button below to continue to the final questionnaires."

  // remove the info parts
    if (condition == 1){
      interventionArray.splice(2,3)
    } else {
      interventionArray.splice(0,1)
       }

    // change the input slides to include what they entered and the option to change it below
    // var start
    // if (condition == 1){start = 2} else {start = 0}
    // for (var i=start;i<interventionArray.length-1;i++){
    //   interventionArray[i]= interventionArray[i] +"     Your answer:   " + data["intervention"][i] + "      If you would like to change your answer, please enter your changes below, otherwise press continue."
      
    // }

  }

  removeChilds(instructions)
 // remove ocean
 $(document.getElementById("ocean")).hide()
 $(document.getElementById("krakenPresent")).hide()

 // create the texts
 // display the texts/questions
if (currentBlock == 7){ // if this is the intervention then display all the info and all the questions and everything
  if (condition == 1){
    // first page
    if (interventionPage == 0){
      instructions.appendChild(createSlider(interventionArray[0], ["clicking as many squares as possible", "re-clicking"]))
    } else if (interventionPage == 1){ // second page
      instructions.appendChild(createSlider(interventionArray[1], ["yes", "no"]))
    } else if (interventionPage == 2 | interventionPage == 3 | interventionPage == 4 | interventionPage == 10){
      instructions.appendChild(createText(interventionArray[interventionPage]))
          // if this is the last intervention page and we are seeing it for the second time then save the data
        if (interventionPage == 10 & currentBlock > 7){
          data["interventionTime"] = interventionTime;
          data["wantChange"] = wantChange;
          var valuesAsJSON = JSON.stringify(data);
          saveData(valuesAsJSON.toString());
         
        }
    } else if (interventionPage > 4 & interventionPage < 10) {
      instructions.appendChild(createOpenQ(interventionArray[interventionPage]))
  
    }
  
  } else {// control condition
    if (interventionPage == 0 | interventionPage == 6){
      instructions.appendChild(createText(interventionArray[interventionPage]))
        // if this is the last intervention page and we are seeing it for the second time then save the data
        if (interventionPage == 6 & currentBlock > 7){
          data["interventionTime"] = interventionTime;
          var valuesAsJSON = JSON.stringify(data);
          saveData(valuesAsJSON.toString());
         
        }
    } else {
      instructions.appendChild(createOpenQ(interventionArray[interventionPage]))
    }
  }


} else { // if this is just the revisit then skip the initial info
  if (condition == 1){
    // first page
    if (interventionPage == 0){
      instructions.appendChild(createSlider(interventionArray[0], ["clicking as many squares as possible", "re-clicking"]))
    } else if (interventionPage == 1){ // second page
      instructions.appendChild(createSlider(interventionArray[1], ["yes", "no"]))
    } else if (interventionPage == interventionArray.length-1){
      instructions.appendChild(createText(interventionArray[interventionPage]))
          // if this is the last intervention page and we are seeing it for the second time then save the data
        if (interventionPage == interventionArray.length-1 & currentBlock > 7){
          data["interventionTime"] = interventionTime;
          data["wantChange"] = wantChange;
          var valuesAsJSON = JSON.stringify(data);
          saveData(valuesAsJSON.toString());
          
        }
    } else {
      instructions.appendChild(createOpenQ(interventionArray[interventionPage]))
  
    }
  
  } else {// control condition
    if (interventionPage == interventionArray.length-1){
      instructions.appendChild(createText(interventionArray[interventionPage]))
        // if this is the last intervention page and we are seeing it for the second time then save the data
        if (interventionPage == interventionArray.length-1 & currentBlock > 7){
          data["interventionTime"] = interventionTime;
          data["wantChange"] = wantChange;
          var valuesAsJSON = JSON.stringify(data);
          saveData(valuesAsJSON.toString());
          
        }
    } else {
      instructions.appendChild(createOpenQ(interventionArray[interventionPage]))
    }
  }


}


}

// create texts
if (condition == 1 ){// intervention

  interventionArray.push( // first page
  "In the game you just played, you will never have all the information about all the squares in the ocean. What do you think is a better strategy: clicking as many squares as possible to find out as much information as possible or re-clicking the squares with the most fish?"
  )
  interventionArray.push(
    "Still in the context of the fishing game you just played, do you think that worrying about whether you found the best square helps you be better at the game?"
  )
  interventionArray.push(
    "As we mentioned in the beginning, people have told us that they tend to worry about not having found the optimal square or that they feel anxious about finding the Kraken. We would like to give people a strategy that might help them not feel this way."
  )
  interventionArray.push(
    "A central aspect of this strategy is that worrying is unnecessary, no matter the content of the worries. This is because all thoughts, no matter their validity, are just thoughts, not facts."
  )
  interventionArray.push(
    "We would like your help in making people understand and use this strategy. We will now try to explain this strategy to you. We hope it makes sense. What we need your help with is to come up with a better explanation that would help others really understand it better than from our own explanations. Please think carefully about your answers and enter them into the corresponding text fields. After you submit the study, we will review your responses and if they are nonsensical, we will not be able to pay you your bonus payment."
  )
  interventionArray.push(
    "How could you explain to someone that it is possible to be well prepared without worrying?"
  )
  interventionArray.push(
    "Studies have shown that worrying is a controllable process. Please give us an example of when you managed to control worries that shows to others that worrying is a controllable process."
  )
  interventionArray.push(
    "If worrying were an effective problem-solving strategy, people who worry more would have fewer problems. Science has shown that the opposite is actually the case and people that worry more report having more problems. What would you say to somebody to convince them of this?"
  )
  interventionArray.push(
    "Imagine a person who worried a lot during the game you just played. Would the person still have been able to solve the game, if they had not worried? How?"
  
  )
  interventionArray.push(
    "One idea for people to detach from their worries is to get them to see their thoughts as just passing mental events rather than facts. What could we say to people who worry in the game to help them see their worries as just passing thoughts? Just telling people to do this doesn't work so well, so we need some more intuitive explanation of why this would be helpful. "
  )
  interventionArray.push(
    "Thank you for your answers! You will now play the game again for 5 more rounds. During the game, please think about your answers and whether you would like to change anything about them. After the 5 rounds of the game, you will be able to change your answers, if you want to." 
  )
   } else {// control intervention
  
  interventionArray.push(
    "As we announced in the beginning, we would like some feedback on how you are finding the game so far.<br><br>Please think carefully about your answers and enter them into the corresponding text fields.<br><br>After you submit the study, we will review your responses and if they are nonsensical, we will not be able to pay you your bonus payment."
  )
  interventionArray.push(
    "How would you explain the game to someone who has never played it before?"
    )
  interventionArray.push(
    "What was your strategy to gain more points?"
  )
  interventionArray.push(
    "What was your strategy to avoid the kraken?"
  )
  interventionArray.push(
   "What did you like about the game?" 
  )
  interventionArray.push(
    "What did you dislike about the game?"
  )
  interventionArray.push(
    "Thank you for your answers! You will now play the game again for 5 more rounds. Afterwards, you will be able to change your answers, if you want to."
  )
   }

// end intervention code ----------------------------------------------



// START

 // Instructions
 var instructions = document.getElementById("instructions");
 var instructionContainer = document.createElement("instructionContainer");
 instructionContainer.setAttribute("id", "instructionContainer");

 var instructionHeading = document.createElement("instructionHeading");
 instructionHeading.innerHTML = "Instructions";
 instructionHeading.setAttribute("id", "instructionHeading");
 instructionContainer.appendChild(instructionHeading);

 var instructionContent = document.createElement("instructionContent");
 instructionContent.setAttribute("id", "instructionContent");
 instructionContainer.appendChild(instructionContent);

 var instructionText = document.createElement("instructionText");
 instructionText.setAttribute("id", "instructionText");

 if (condition == 1){// if they are in the experimental condition
 instructionText.innerHTML = "In this study, we would like to get some help from you in designing an intervention, that will make people less anxious. " +
 "More precisely: We designed a game that often makes people feel anxious and makes them worry. We would now like to help people be less anxious, worry less and thereby do better in the game. <br><br>"+
 "For you to be able to help us in designing this intervention, you first need to know <b>what the game is about</b>. You will therefore first get a <b>tutorial</b> on the game and <b>play a couple of rounds of the game</b> before we <b>tell you</b> more <b>about the strategy</b> we would like to convey.<br><br>"+
"After helping us design the instructions for the strategy, you will get to <b>play the game again</b> to see <b>whether the strategy fits the game</b>. If you realise at this point, that you are not happy with the way you designed the strategy, you will have the opportunity to <b>change it</b> after you finished playing the game for the second time.<br><br>" +
"To make your experience of the game exactly the way our future participants will experience it, you will also be able to get a <b>bonus payment</b> depending on how well you do in the game, so try to do your best.<br><br>"+
"We will also ask you from time to time, <b>how nervous</b> you felt during the previous round, to gather more data on what aspects of the game make people nervous. Please answers this <b>honestly</b> and not based on what you think we would like to hear!<br><br>"+
"In the end, we will also ask you some questions to get to know you a bit better.<br><br>"+
"Let's get started with the game tutorial!"//KW: first instructions

 } else { // if they are in the control condition
  instructionText.innerHTML = "In this study you will be playing a novel game that we designed quite recently. " +
  "Since it is so new, we don't know whether people will like and what kind of strategies they will use when playing the game.<br><br>"+
  "We would therefore like for you to be one of our <b>test subjects</b>. You will receive a tutorial for the game, then play 6 rounds to get familiar with it.<br><br> "+
  "After that we will ask you some questions about how you liked the game. After that, you will play another 5 rounds of the game so that you can see, whether having more experience changes how you feel about the game.<br><br>"+
  "You will then have the possibility to change your answers, if playing for more rounds changed your mind on some of them. " +
  "To make your experience of the game exactly the way our future participants will experience it, you will also be able to get a <b>bonus payment</b> depending on how well you do in the game, so try to do your best.<br><br>"+
  "Some people have told us that the game made them feel nervous. We will therefore ask you from time to time <b>how nervous</b> you felt during the previous round. Please answer this <b>honestly</b> and not based on what you think we would like to hear!<br><br>"
"In the end, we will also ask you some questions to get to know you a bit better.<br><br>"+
"Let's get started with the game tutorial!"//KW: first instructions
 }


 // KW: create arrow to continue
 var proceed = document.createElement("button");
 proceed.setAttribute("id", "first");
 proceed.innerHTML = "Click to continue<br><br>";
 var proceedImg = document.createElement("arrowImg");
 proceedImg.innerHTML = "<img src='assets/arrow.png'>";
 proceedImg.setAttribute("id", "arrowImg");
 proceed.appendChild(proceedImg);
 proceed.addEventListener("click", init)

 instructions.innerHTML = "<h1>Welcome to the main part of the study!</h1>"
 instructionContent.appendChild(instructionText);
 instructionContent.appendChild(proceed);
 instructions.appendChild(instructionContainer)

 // don't show the ocean yet
 $(document.getElementById("ocean")).hide()


//alert
// var bsAlert = function(message) {
//  if ($("#bs-alert").length == 0) {
//     $('body').append('<div class="modal tabindex="-1" id="bs-alert">'+
//     '<div class="modal-dialog">'+
//       '<div class="modal-content">'+
//         '<div class="modal-header">'+
//           '<button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button>'+
//           '<h4 class="modal-title">Alert</h4>'+
//         '</div>'+
//         '<div class="modal-body">'+
//             message+
//         '</div>'+
//         '<div class="modal-footer">'+
//           '<button type="button" class="btn btn-default" data-dismiss="modal">Ok</button>'+
//         '</div>'+
//       '</div>'+
//     '</div>'+
//   '</div>')
//   } else {
//       $("#bs-alert .modal-body").text(message);
//   }    
//   $("#bs-alert").modal();
// }
// window.alert = bsAlert; 