
    // Your web app's Firebase configuration
    var firebaseConfig = {
      apiKey: "AIzaSyCwv1H_4608k_iQN6-HHVI2wxrTvxMjvKc",
      authDomain: "firestore-1add2.firebaseapp.com",
      databaseURL: "https://firestore-1add2.firebaseio.com",
      projectId: "firestore-1add2",
      storageBucket: "firestore-1add2.appspot.com",
      messagingSenderId: "410516002311",
      appId: "1:410516002311:web:0c1e7cc41724dca1db3ca2",
      measurementId: "G-S7ZZZRVQX7"
    };
    // Initialize Firebase
    firebase.initializeApp(firebaseConfig);
    var firestore = firebase.firestore();

    const docRef = firestore.doc("robot/sky");
    const docRef_control = firestore.doc("robot/control");
    const outputHeader = document.querySelector("#NumberOfDroneOutput");//불러오기
    const outputHeader2 = document.querySelector("#timedate"); //불러오기

    getRealtimeUpdates = function() {
      docRef.onSnapshot(function (doc) {
        if (doc && doc.exists) {
          const myData = doc.data();
          outputHeader.innerText = "Number of Drone: " + myData.Num_of_drone;
          outputHeader2.innerText = "Date: " + myData.dates;
        }
      });
    }

    getRealtimeUpdates();
