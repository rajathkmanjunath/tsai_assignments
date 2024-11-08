function injectFloatingWindow() {
  fetch(chrome.runtime.getURL('floating-window.html'))
    .then(response => response.text())
    .then(data => {
      const div = document.createElement('div');
      div.innerHTML = data;
      document.body.appendChild(div);
      
      initializeFloatingWindow();
    });
}

function initializeFloatingWindow() {
  const floatingWindow = document.getElementById('floating-window');
  const videoSpeedBtn = document.getElementById('video-speed-btn');
  const speedControls = document.getElementById('speed-controls');
  const setSpeedBtn = document.getElementById('set-speed');
  const resetSpeedBtn = document.getElementById('reset-speed');
  const increaseSpeedBtn = document.getElementById('increase-speed');
  const decreaseSpeedBtn = document.getElementById('decrease-speed');
  const emailCleanupBtn = document.getElementById('email-cleanup-btn');
  const handle = document.getElementById('handle');

  let isDragging = false;
  let currentX;
  let currentY;
  let initialX;
  let initialY;
  let xOffset = 0;
  let yOffset = 0;

  videoSpeedBtn.addEventListener('click', () => {
    speedControls.style.display = speedControls.style.display === 'none' ? 'block' : 'none';
    floatingWindow.classList.toggle('expanded');
  });

  setSpeedBtn.addEventListener('click', () => {
    setSpeed(2);
    compressWindow();
  });
  resetSpeedBtn.addEventListener('click', () => {
    resetSpeed();
    compressWindow();
  });
  increaseSpeedBtn.addEventListener('click', () => {
    increaseSpeed();
    compressWindow();
  });
  decreaseSpeedBtn.addEventListener('click', () => {
    decreaseSpeed();
    compressWindow();
  });

  handle.addEventListener('mousedown', dragStart);
  document.addEventListener('mousemove', drag);
  document.addEventListener('mouseup', dragEnd);

  // Check if we're on Gmail and show the email cleanup button if appropriate
  if (window.location.hostname === 'mail.google.com') {
    emailCleanupBtn.classList.remove('hidden');
    setInterval(checkForGmailDraft, 1000);
  }

  emailCleanupBtn.addEventListener('click', cleanupEmail);

  function dragStart(e) {
    initialX = e.clientX - xOffset;
    initialY = e.clientY - yOffset;

    if (e.target === handle) {
      isDragging = true;
    }
  }

  function drag(e) {
    if (isDragging) {
      e.preventDefault();
      currentX = e.clientX - initialX;
      currentY = e.clientY - initialY;

      xOffset = currentX;
      yOffset = currentY;

      setTranslate(currentX, currentY, floatingWindow);
    }
  }

  function dragEnd(e) {
    initialX = currentX;
    initialY = currentY;

    isDragging = false;
  }

  function setTranslate(xPos, yPos, el) {
    el.style.transform = `translate3d(${xPos}px, ${yPos}px, 0)`;
  }

  function compressWindow() {
    speedControls.style.display = 'none';
    floatingWindow.classList.remove('expanded');
  }

  function checkForGmailDraft() {
    const composeBox = document.querySelector('.Am.Al.editable.LW-avf');
    if (composeBox) {
      emailCleanupBtn.style.display = 'block';
    } else {
      emailCleanupBtn.style.display = 'none';
    }
  }
}

function setSpeed(speed) {
  const videos = document.getElementsByTagName('video');
  for (let video of videos) {
    video.playbackRate = speed;
  }
}

function resetSpeed() {
  setSpeed(1);
}

function increaseSpeed() {
  const videos = document.getElementsByTagName('video');
  for (let video of videos) {
    video.playbackRate += 0.2;
  }
}

function decreaseSpeed() {
  const videos = document.getElementsByTagName('video');
  for (let video of videos) {
    video.playbackRate -= 0.2;
  }
}

function cleanupEmail() {
  const composeBox = document.querySelector('.Am.Al.editable.LW-avf');
  if (composeBox) {
    const originalText = composeBox.innerText;
    
    chrome.runtime.sendMessage({action: "cleanupEmail", text: originalText}, response => {
      if (response.cleanedText) {
        composeBox.innerHTML = response.cleanedText;
      } else if (response.error) {
        console.error(response.error);
        alert(response.error);
      }
    });
  }
}

injectFloatingWindow();
