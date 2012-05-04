#include <Background.h>

// bgf::AbelianBgfFactory::f
// holds f_2 * i
// holds even volumes,

const double bgf::AbelianBgfFactory::f[3][31] = 
  {
    // s = - 1
    { 0.06547617944229099, // L = 4
      0.02909035966188364, // L = 6
      0.01636266709965179, // L = 8
      0.01047201857745911, // etc...
      0.007272217235243609,
      0.005342848734259618,
      0.004090617038636703,
      0.003232091910810135,
      0.002617994214431594,
      0.002163631476772091,
      0.001818051398054947,
      0.001549108857162432,
      0.001335711194156749,
      0.001163552854353917,
      0.001022653871123783,
      0.0009058802427982062,
      0.0008080228073446312,
      0.0007252060641493027,
      0.0006544984721263084,
      0.0005936494073082277,
      0.0005409078273801031,
      0.000494894873954382,
      0.0004545128267737379,
      0.0004188790210298618,
      0.0003872772013051997,
      0.0003591212455879561,
      0.0003339277908094847,
      0.0003112953483509835,
      0.0002908882088195577,
      0.0002724239208289968,
      0.0002556634647455237
    },
    { // s = 0
      0.06544984694978736, // L = 4
      0.02908882086657216, // L = 6
      0.01636246173744684, // ...
      0.01047197551196598,
      0.00727220521664304,
      0.005342844648962233,
      0.00409061543436171,
      0.003232091207396907,
      0.002617993877991494,
      0.002163631304125202,
      0.00181805130416076,
      0.001549108803545263,
      0.001335711162240558,
      0.001163552834662886,
      0.001022653858590427,
      0.000905880234599133,
      0.0008080228018492267,
      0.0007252060603854555,
      0.0006544984694978736,
      0.0005936494054402482,
      0.0005409078260313005,
      0.000494894872966256,
      0.00045451282604019,
      0.0004188790204786391,
      0.0003872772008863157,
      0.000359121245266323,
      0.0003339277905601396,
      0.0003112953481559446,
      0.0002908882086657216,
      0.0002724239207067112,
      0.0002556634646476069
    },
    { // s = 1
      0.06531764861843085,
      0.02908111705368769,
      0.01636143443413726,
      0.0104717601382444,
      0.007272145117058098,
      0.00534282422122086,
      0.004090607412689904,
      0.003232087690247792,
      0.002617992195764523,
      0.002163630440881357,
      0.001818050834686175,
      0.001549108535457889,
      0.001335711002658926,
      0.001163552736207413,
      0.001022653795923492,
      0.0009058801936036849,
      0.0008080227743721602,
      0.0007252060415661951,
      0.0006544984563556855,
      0.0005936493961003423,
      0.0005409078192872824,
      0.0004948948680256231,
      0.0004545128223724486,
      0.0004188790177225243,
      0.0003872771987918952,
      0.0003591212436581567,
      0.0003339277893134134,
      0.00031129534718075,
      0.0002908882078965407,
      0.0002724239200952831,
      0.0002556634641580227}
  };